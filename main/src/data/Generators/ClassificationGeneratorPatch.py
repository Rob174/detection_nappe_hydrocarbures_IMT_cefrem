"""Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches and
filter them on the fly (slow if many patches to exclude)"""
import random
from typing import Optional, List, Union, Dict, Tuple

import numpy as np

from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
from main.src.data.Augmentation.Augmenters.NoAugmenter import NoAugmenter
from main.src.data.BatchMaker.BatchMaker import BatchMaker
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.Fabrics.FabricPreprocessedCache import FabricPreprocessedCache
from main.src.data.Datasets.ImageDataset import ImageDataset
from main.src.data.LabelModifier.LabelModifierFactory import LabelModifierFactory
from main.src.data.MarginCheck import MarginCheck
from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.data.balance_classes.BalanceClassesNoOther import BalanceClassesNoOther
from main.src.data.balance_classes.BalanceClassesOnlyOther import BalanceClassesOnlyOther
from main.src.data.balance_classes.BalanceNoBalance import BalanceNoBalance
from main.src.enums import EnumAugmenter, EnumBalance, EnumLabelModifier, EnumClasses, EnumDataset
from main.src.param_savers.BaseClass import BaseClass


class ClassificationGeneratorPatch(BaseClass):
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

    def __init__(self, input_size: int = None,
                 limit_num_images: int = None, balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor: int = 100, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent=0.7, grid_size_px: int = 1000, threshold_margin: int = 1000,
                 tr_batch_size: int = 10, valid_batch_size: int = 100):
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_global_name = "attr_dataset"
        self.attr_image_dataset, self.attr_label_dataset, self.dico_infos = FabricPreprocessedCache()()
        self.attr_grid_size_px = grid_size_px
        self.attr_limit_num_images = limit_num_images
        self.attr_check_margin_reject = MarginCheck(threshold=threshold_margin)
        self.attr_augmentation_factor = augmentation_factor
        self.datasets = {
            "tr": list(self.attr_image_dataset.keys())[:int(len(self.attr_image_dataset) * tr_percent)],
            "valid": list(self.attr_image_dataset.keys())[int(len(self.attr_image_dataset) * tr_percent):],
            "all": list(self.attr_image_dataset.keys())
        }
        self.attr_tr_batch_maker = BatchMaker(batch_size=tr_batch_size, num_elems_gen=4)
        self.attr_valid_batch_maker = BatchMaker(batch_size=valid_batch_size, num_elems_gen=4)
        self.batch_makers = {
            "tr": self.attr_tr_batch_maker,
            "valid": self.attr_valid_batch_maker,
            "all": BatchMaker(batch_size=1, num_elems_gen=4)
        }
        self.attr_global_name = "attr_dataset"
        self.attr_label_modifier = LabelModifierFactory().create(label_modifier, self.attr_label_dataset.attr_mapping,
                                                                 classes_to_use)

        if balance == EnumBalance.NoBalance:
            self.attr_balance = BalanceNoBalance()
        elif balance == EnumBalance.BalanceClasses1:
            # see class DataSentinel1Segmentation for documentation on attr_class_mapping storage and access to values
            self.attr_balance = BalanceClassesNoOther(other_index=self.attr_label_dataset.attr_mapping["other"])
        elif balance == EnumBalance.BalanceClasses2:
            self.attr_balance = BalanceClassesOnlyOther(other_index=self.attr_label_dataset.attr_mapping["other"])
        else:
            raise NotImplementedError
        if augmentations_img != "none":
            if augmenter_img == EnumAugmenter.Augmenter1:
                self.attr_augmenter = Augmenter1(allowed_transformations=[augmentations_img],
                                                 patch_size_before_final_resize=
                                                 self.attr_grid_size_px,
                                                 patch_size_final_resize=input_size
                                                 )

            else:
                self.attr_augmenter = NoAugmenter(
                                                  patch_size_before_final_resize=self.attr_grid_size_px,
                                                  patch_size_final_resize=input_size
                                                  )
        else:
            raise NotImplementedError(f"{augmenter_img} is not implemented")
        # Cache to store between epochs rejected images if we have no image augmenter
        self.cache_img_id_rejected = []

    def set_datasets(self, image_dataset: ImageDataset, label_dataset: AbstractDataset, dico_infos: Dict):
        """Change the origin of the patches

        Args:
            image_dataset: ImageDataset
            label_dataset: AbstractDataset Points or Images
            dico_infos: Dict, containing for each id of image the source image (under key source_img) and the transformation matrix (under key transformation_matrix) applied to get the patch
        Returns:

        """
        self.attr_image_dataset = image_dataset
        self.attr_label_dataset = label_dataset
        self.dico_infos = dico_infos

    def __iter__(self, dataset: Union[EnumDataset, List[str]] = EnumDataset.Train):
        if isinstance(dataset, list):
            keys = dataset
            batch_maker = self.batch_makers["all"]
        else:
            keys = self.datasets[dataset]
            batch_maker = self.batch_makers[dataset]
        return iter(batch_maker.batch(self.generator(keys)))

    def generator(self, images_available):
        """

        Args:

        Returns:
            generator of the attr_dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the attr_model ;
                   classif: np.ndarray (shape (num_classes,), Generators patch ;
                   transformation_matrix:  the transformation matrix to transform the source image
                   item: str name of the source image
        """
        for num_dataset in range(self.attr_augmentation_factor):
            random.shuffle(images_available)
            for item in images_available:
                image = self.attr_image_dataset.get(item)
                polygons = self.attr_label_dataset.get(item)
                partial_transformation_matrix = self.attr_augmenter.choose_new_augmentations(image)
                for patch_upper_left_corner_coords in np.random.permutation(
                        self.attr_augmenter.get_grid(image.shape, partial_transformation_matrix)):
                    annotations_patch, transformation_matrix = self.attr_augmenter.transform_label(
                        data=polygons,
                        partial_transformation_matrix=partial_transformation_matrix,
                        patch_upper_left_corner_coords=patch_upper_left_corner_coords
                    )
                    # Create the Generators label with the proper technic
                    classification = self.attr_label_modifier.make_classification_label(annotations_patch)
                    balance_reject = self.attr_balance.filter(self.attr_label_modifier.get_initial_label())
                    # if balance_reject is True:
                    #     continue
                    image_patch, transformation_matrix = self.attr_augmenter.transform_image(
                        image=image,
                        partial_transformation_matrix=partial_transformation_matrix,
                        patch_upper_left_corner_coords=patch_upper_left_corner_coords
                    )
                    reject = self.attr_check_margin_reject.check_reject(image_patch)
                    # if reject is True:
                    #     continue
                    # convert the image to rgb (as required by pytorch): not ncessary the best transformation as we multiply by 3 the amount of data
                    image_patch = np.stack((image_patch,) * 3, axis=0)
                    yield image_patch, classification, transformation_matrix, item


    def get_patch(self, image: np.ndarray, annotation: np.ndarray, patch_upper_left_corner_coords: Tuple[int, int],
                  standardizer: AbstractStandardizer, transformation_matrix: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate image patch and corresponding annotation for the given parameters

        Args:
            image: np.ndarray, image on which to apply the transformations
            annotation: np.ndarray, annotation on which to apply the transformations
            patch_upper_left_corner_coords: coordinates of the upper left corner to get
            standardizer: object giving allowing to standardize the patch
            transformation_matrix: Optional[np.ndarray] (3,3) transformation matrix to apply to the image and annotation ⚠️⚠️ no rotation allowed as we have an array annotation

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]
            - image_patch: patch extracted from the image
            - Generators: vector containing true probabilities of presence of annotation
            - transformation_matrix: 3,3 transformation matrix applied
        """
        if transformation_matrix is None:
            transformation_matrix = np.identity(3)
        image_patch, transformation_matrix = self.attr_augmenter.transform_image(
            image=image,
            partial_transformation_matrix=transformation_matrix,
            patch_upper_left_corner_coords=patch_upper_left_corner_coords
        )
        annotation_patch, transformation_matrix = self.attr_augmenter.transform_image(
            image=annotation,
            partial_transformation_matrix=transformation_matrix,
            patch_upper_left_corner_coords=patch_upper_left_corner_coords
        )
        classification = self.attr_label_modifier.make_classification_label(annotation_patch)
        image_patch = np.stack((image_patch,) * 3, axis=0)
        image_patch = standardizer.standardize(image_patch)
        return image_patch, classification, transformation_matrix

    def __len__(self):
        return None

    def set_standardizer(self, standardizer: AbstractStandardizer):
        self.attr_standardizer = standardizer

    def len(self, dataset: str) -> Optional[int]:
        return None
