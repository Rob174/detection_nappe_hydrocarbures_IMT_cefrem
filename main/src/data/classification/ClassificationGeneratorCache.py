"""Class that takes inputs from the filtered hdf5 file to create a new dataset. Far quicker than the original ClassificationGeneratorCache."""

import json
import random
from typing import Tuple, Generator, Optional, Dict

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.Augmentation.Augmentations.AugmentationApplier.AugmentationApplierImage import \
    AugmentationApplierImage
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache
from main.src.data.Datasets.ImageDataset import ImageDataset
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.GridMaker.GridMaker import GridMaker
from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.data.TwoWayDict import TwoWayDict
from main.src.data.classification.ClassificationGeneratorPatch import ClassificationGeneratorPatch
from main.src.data.PatchAdder.NoClassPatchAdder import NoClassPatchAdder
from main.src.data.PatchAdder.OtherClassPatchAdder import OtherClassPatchAdder
from main.src.data.PatchAdder.PatchAdderCallback import PatchAdderCallback
from main.src.data.Standardizer.StandardizerCacheMixed import StandardizerCacheMixed
from main.src.data.Standardizer.StandardizerCacheSeepSpill import StandardizerCacheSeepSpill
from main.src.enums import EnumLabelModifier, EnumClassPatchAdder
from main.src.enums import EnumClasses
from main.src.param_savers.BaseClass import BaseClass


class ClassificationGeneratorCache(BaseClass):
    """Class that takes inputs from the filtered hdf5 file to create a new dataset

    Args:

        label_modifier: EnumLabelModifier
        classes_to_use: Tuple[EnumClasses] classes to supply to the LabelModifier
        tr_percent: float percentage of images allocated to training
        limit_num_images: int limit the number of images for the training dataset
        other_class_adder: EnumClassPatchAdder to select classadder object
        interval: int, interval between two un annotated patches
    """

    def __init__(self,
                 label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent: float = 0.7, limit_num_images: int = None,
                 other_class_adder: EnumClassPatchAdder = EnumClassPatchAdder.NoClassPatchAdder,
                 interval: int = 1):
        print("Using ClassificationGeneratorCache")
        self.attr_standardization = True
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_limit_num_images = limit_num_images
        self.attr_image_dataset, self.attr_label_dataset,self.dico_infos = FabricFilteredCache()()
        self.tr_keys = list(self.attr_image_dataset.keys())[:int(len(self.attr_image_dataset) * tr_percent)]
        if self.attr_limit_num_images is not None:
            self.tr_keys = self.tr_keys[:self.attr_limit_num_images]
        self.valid_keys = list(self.attr_image_dataset.keys())[int(len(self.attr_image_dataset) * tr_percent):]
        self.attr_global_name = "attr_dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = LabelModifier0(class_mapping=self.attr_label_dataset.attr_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=self.attr_label_dataset.attr_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=self.attr_label_dataset.attr_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")

        if other_class_adder == EnumClassPatchAdder.OtherClassPatchAdder:
            self.attr_other_class_adder = OtherClassPatchAdder(interval=interval)
            self.attr_standardizer = StandardizerCacheMixed(interval=interval)
        elif other_class_adder == EnumClassPatchAdder.NoClassPatchAdder:
            self.attr_other_class_adder = NoClassPatchAdder(interval=interval)
            self.attr_standardizer = StandardizerCacheSeepSpill()
        self.attr_patch_adder_callback = PatchAdderCallback(class_adders=[self.attr_other_class_adder])
    def set_datasets(self,image_dataset: ImageDataset, label_dataset: AbstractDataset, dico_infos: Dict):
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
    def __iter__(self, dataset="tr"):
        return iter(self.generator())

    def generator(self, dataset="tr") -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, str], None, None]:
        """Generator that generates data for the trainer

        Args:
            dataset: str, tr or valid to choose source images for tr or valid attr_dataset

        Returns:
            generator of the attr_dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the attr_model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   transformation_matrix:  the transformation matrix to transform the source image
                   item: str name of the source image
        """
        images_available = self.tr_keys if dataset == "tr" else self.valid_keys
        random.shuffle(images_available)
        self.attr_patch_adder_callback.on_epoch_start()
        for id in images_available:
            data = self.attr_other_class_adder.generate_if_required()
            if data is not None:
                image, annotation, transformation_matrix, source_img = data
                image, annotation = self.process_infos(image, annotation)
                yield image, annotation, transformation_matrix, source_img
            # Open image and annotations
            image = self.attr_image_dataset.get(id)
            annotation = self.attr_label_dataset.get(id)
            source_img = self.dico_infos[id]["source_img"]
            transformation_matrix = np.array(self.dico_infos[id]["transformation_matrix"])
            image, annotation = self.process_infos(image, annotation)
            yield image, annotation, transformation_matrix, source_img
    def process_infos(self, image, annotation):
        image = self.attr_standardizer.standardize(image)
        image = np.stack((image,) * 3, axis=0)
        annotation = self.attr_label_modifier.make_classification_label(annotation)
        return image, annotation
    def get_patch(self,image: np.ndarray,annotation: np.ndarray,
                  patch_size_after_resize: int,
                  patch_upper_left_corner_coords: Tuple[int,int],
                  transformation_matrix: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Generate image patch and corresponding annotation for the given parameters

        Args:
            image: np.ndarray, image on which to apply the transformations
            annotation: np.ndarray, annotation on which to apply the transformations
            patch_upper_left_corner_coords: coordinates of the upper left corner to get
            transformation_matrix: Optional[np.ndarray] (3,3) transformation matrix to apply to the image and annotation ⚠️⚠️ no rotation allowed as we have an array annotation

        Returns:
            Tuple[np.ndarray,np.ndarray,np.ndarray]
            - image_patch: patch extracted from the image
            - classification: vector containing true probabilities of presence of annotation
            - transformation_matrix: 3,3 transformation matrix applied
        """
        if transformation_matrix is None:
            transformation_matrix = np.identity(3)
        applier = AugmentationApplierImage(grid_maker=GridMaker(patch_size_final_resize=patch_size_after_resize),
                                           patch_size_final_resize=patch_size_after_resize)
        image_patch, transformation_matrix = applier.transform(
            data=image,
            partial_transformation_matrix=transformation_matrix,
            patch_upper_left_corner_coords=patch_upper_left_corner_coords
        )
        annotation_patch, transformation_matrix = applier.transform(
            data=annotation,
            partial_transformation_matrix=transformation_matrix,
            patch_upper_left_corner_coords=patch_upper_left_corner_coords
        )
        image_patch,classification = self.process_infos(image,annotation)
        return image_patch,classification,transformation_matrix


    def len(self, dataset) -> Optional[int]:
        if dataset == "tr":
            return len(self.tr_keys)
        else:
            return len(self.valid_keys)
