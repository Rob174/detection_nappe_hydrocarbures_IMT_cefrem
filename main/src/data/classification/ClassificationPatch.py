"""Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches and
filter them"""
import json
import random
from typing import Optional, List, Union
from typing import Tuple

import numpy as np
from h5py import File
from rasterio.transform import Affine, rowcol

from main.FolderInfos import FolderInfos
from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
from main.src.data.Augmentation.Augmenters.NoAugmenter import NoAugmenter
from main.src.data.TwoWayDict import TwoWayDict
from main.src.enums import EnumAugmenter
from main.src.data.balance_classes.BalanceClasses1 import BalanceClasses1
from main.src.data.balance_classes.BalanceClasses2 import BalanceClasses2
from main.src.enums import EnumBalance
from main.src.data.balance_classes.no_balance import NoBalance
from main.src.data.classification.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.classification.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.classification.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.classification.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.enums import EnumLabelModifier
from main.src.enums import EnumClasses
from main.src.data.patch_creator.MarginCheck import MarginCheck
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSegmentation import DataSentinel1Segmentation
from main.src.data.segmentation.PointAnnotations import PointAnnotations
from main.src.enums import EnumDataset
from main.src.param_savers.BaseClass import BaseClass


class ClassificationPatch(BaseClass):
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

    attr_original_class_mapping = TwoWayDict(  # a twoway dict allowing to store pairs of hashable objects:
        {  # Formatted in the following way: src_index in cache, name, the position encode destination index
            0: "other",
            1: "seep",
            2: "spill",
        })
    def __init__(self, input_size: int = None,
                 limit_num_images: int = None, balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor: int = 100, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent=0.7, grid_size_px: int = 1000, threshold_margin:int = 1000):
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_global_name = "attr_dataset"
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            self.images_infos = json.load(fp)
        with open(f"{FolderInfos.input_data_folder}pixel_stats.json", "r") as fp:
            self.pixel_stats = json.load(fp)
        self.images = File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r")
        self.attr_class_mapping = TwoWayDict(
            {k: v for k, v in DataSentinel1Segmentation.attr_original_class_mapping.items()})
        self.attr_grid_size_px = grid_size_px
        self.attr_limit_num_images = limit_num_images
        self.attr_check_margin_reject = MarginCheck(threshold=threshold_margin)
        self.attr_augmentation_factor = augmentation_factor
        super(ClassificationPatch, self).__init__(limit_num_images, input_size=input_size, )
        self.datasets = {
            "tr":list(self.images.keys())[:int(len(self.images) * tr_percent)],
            "valid":list(self.images.keys())[int(len(self.images) * tr_percent):]
        }
        self.attr_global_name = "attr_dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = LabelModifier0(class_mapping=ClassificationPatch.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=ClassificationPatch.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=ClassificationPatch.attr_original_class_mapping)
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
                                                     self.attr_grid_size_px,
                                                     patch_size_final_resize=input_size,
                                                     label_access_function=self.annotations_labels.get
                                                     )

            else:
                self.attr_img_augmenter = NoAugmenter(allowed_transformations=augmentations_img,
                                                     patch_size_before_final_resize=
                                                     self.attr_grid_size_px,
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
        for num_dataset in range(self.attr_augmentation_factor):
            random.shuffle(images_available)
            for item in images_available:
                image = self.images[item]
                image = np.array(image, dtype=np.float32)
                partial_transformation_matrix = self.attr_img_augmenter.choose_new_augmentations(image)
                for patch_upper_left_corner_coords in np.random.permutation(self.attr_img_augmenter.get_grid(image.shape, partial_transformation_matrix)):
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
                    reject = self.attr_check_margin_reject.check_reject(image_patch)
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
