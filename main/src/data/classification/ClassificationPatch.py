"""Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches and
filter them"""

import random
from typing import Optional, List, Union
from typing import Tuple

import numpy as np
from rasterio.transform import Affine, rowcol

from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
from main.src.data.Augmentation.Augmenters.NoAugmenter import NoAugmenter
from main.src.data.Augmentation.Augmenters.enums import EnumAugmenter
from main.src.data.balance_classes.BalanceClasses1 import BalanceClasses1
from main.src.data.balance_classes.BalanceClasses2 import BalanceClasses2
from main.src.data.balance_classes.enums import EnumBalance
from main.src.data.balance_classes.no_balance import NoBalance
from main.src.data.classification.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.classification.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.classification.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.classification.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSegmentation import DataSentinel1Segmentation
from main.src.data.segmentation.PointAnnotations import PointAnnotations
from main.src.training.enums import EnumDataset


class ClassificationPatch(DataSentinel1Segmentation):
    """Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches and
    filter them.

    Args:
        patch_creator: the object of PatchCreator0 class managing patches
        input_size: the size of the image provided as input to the attr_model ⚠️
        limit_num_images: limit the number of image in the attr_dataset per epoch (before filtering)
        balance: EnumBalance indicating the class used to balance images
        augmentations_img: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_img: opt EnumAugmenter, name of the augmenter to use on source image
        augmentation_factor: the number of replicas of the original attr_dataset to do
        label_modifier: EnumLabelModifier
    """

    def __init__(self, patch_creator: Patch_creator0, input_size: int = None,
                 limit_num_images: int = None, balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor: int = 100, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent=0.7):
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.patch_creator = patch_creator
        self.attr_limit_num_images = limit_num_images
        self.attr_resizer = Resizer(out_size_w=input_size)
        self.attr_augmentation_factor = augmentation_factor
        super(ClassificationPatch, self).__init__(limit_num_images, input_size=input_size, )
        self.datasets = {
            "tr":list(self.images.keys())[:int(len(self.images) * tr_percent)],
            "valid":list(self.images.keys())[int(len(self.images) * tr_percent):]
        }
        self.attr_global_name = "attr_dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = LabelModifier0(class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")
        if balance == EnumBalance.NoBalance:
            self.attr_balance = NoBalance()
        elif balance == EnumBalance.BalanceClasses1:
            # see class DataSentinel1Segmentation for documentation on attr_class_mapping storage and access to values
            self.attr_balance = BalanceClasses1(other_index=self.attr_original_class_mapping["other"])
        elif balance == EnumBalance.BalanceClasses2:
            self.attr_balance = BalanceClasses2(other_index=self.attr_original_class_mapping["other"])
        else:
            raise NotImplementedError
        if augmentations_img != "none":
            self.annotations_labels = PointAnnotations()
            if augmenter_img == EnumAugmenter.Augmenter1:
                self.attr_img_augmenter = Augmenter1(allowed_transformations=augmentations_img,
                                                     patch_size_before_final_resize=
                                                     self.patch_creator.attr_grid_size_px,
                                                     patch_size_final_resize=input_size,
                                                     label_access_function=self.annotations_labels.get
                                                     )

            else:
                self.attr_img_augmenter = NoAugmenter(allowed_transformations=augmentations_img,
                                                     patch_size_before_final_resize=
                                                     self.patch_creator.attr_grid_size_px,
                                                     patch_size_final_resize=input_size,
                                                     label_access_function=self.annotations_labels.get
                                                      )
        else:
            raise NotImplementedError(f"{augmenter_img} is not implemented")
        # Cache to store between epochs rejected images if we have no image augmenter
        self.cache_img_id_rejected = []

    def __iter__(self, dataset: Union[EnumDataset,List[str]] = EnumDataset.Train):
        if isinstance(dataset,list):
            keys = dataset
        else:
            keys = self.datasets[dataset]
        return iter(self.generator(keys))

    def generator(self, images_available):
        """

        Args:

        Returns:
            generator of the attr_dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the attr_model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   transformation_matrix:  the transformation matrix to transform the source image
                   item: str name of the source image
        """
        if isinstance(self.attr_img_augmenter, Augmenter1) is False:
            raise Exception("Only augmenter1 is supported with this method of attr_dataset generation")
        for num_dataset in range(self.attr_augmentation_factor):
            random.shuffle(images_available)
            for item in images_available:
                image = self.images[item]
                image = np.array(image, dtype=np.float32)
                partial_transformation_matrix = np.array([[256/1000,0,0],[0,256/1000,0],[0,0,1]],dtype=np.float32)#self.attr_img_augmenter.choose_new_augmentations(image)
                for patch_upper_left_corner_coords in self.attr_img_augmenter.get_grid(image.shape, partial_transformation_matrix):#np.random.permutation(self.attr_img_augmenter.get_grid(image.shape, partial_transformation_matrix)):
                    annotations_patch, transformation_matrix = self.attr_img_augmenter.transform_label(
                        image_name=item,
                        partial_transformation_matrix=partial_transformation_matrix,
                        patch_upper_left_corner_coords=patch_upper_left_corner_coords
                    )
                    # Create the classification label with the proper technic ⚠️⚠️ inheritance
                    classification = self.attr_label_modifier.make_classification_label(annotations_patch)
                    balance_reject = self.attr_balance.filter(self.attr_label_modifier.get_initial_label())
                    if balance_reject is True:
                        continue
                    image_patch, transformation_matrix = self.attr_img_augmenter.transform_image(
                        image=image,
                        partial_transformation_matrix=partial_transformation_matrix,
                        patch_upper_left_corner_coords=patch_upper_left_corner_coords
                    )
                    reject = self.patch_creator.check_reject(image_patch, threshold_px=10)
                    if reject is True:
                        continue
                    # convert the image to rgb (as required by pytorch): not ncessary the best transformation as we multiply by 3 the amount of data
                    image_patch = np.stack((image_patch,)*3, axis=0)  # 0 ns most of the time
                    yield image_patch, classification, transformation_matrix, item


    def __len__(self):
        return None
    def set_standardizer(self, standardizer: AbstractStandardizer):
        self.attr_standardizer = standardizer
    def set_annotator(self,annotations):
        self.annotations_labels = annotations

    def len(self, dataset: str) -> Optional[int]:
        return None
