"""Class that takes inputs from the filtered hdf5 file to create a new dataset. Far quicker than the original ClassificationCache."""

import json
import random
from typing import Tuple, Generator, Optional

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.PatchAdder.NoClassPatchAdder import NoClassPatchAdder
from main.src.data.PatchAdder.OtherClassPatchAdder import OtherClassPatchAdder
from main.src.data.PatchAdder.PatchAdderCallback import PatchAdderCallback
from main.src.data.Standardizer.StandardizerCacheMixed import StandardizerCacheMixed
from main.src.data.Standardizer.StandardizerCacheSeepSpill import StandardizerCacheSeepSpill
from main.src.enums import EnumLabelModifier, EnumClassPatchAdder
from main.src.enums import EnumClasses
from main.src.param_savers.BaseClass import BaseClass


class ClassificationCache(BaseClass):
    """Class that takes inputs from the filtered hdf5 file to create a new dataset

    Args:

        label_modifier: EnumLabelModifier
        classes_to_use: Tuple[EnumClasses] classes to supply to the LabelModifier
        tr_percent: float percentage of images allocated to training
        limit_num_images: int limit the number of images for the training dataset
        other_class_adder: EnumClassPatchAdder to select classadder object
        interval: int, interval between two un annotated patches
    """

    def __init__(self, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent: float = 0.7, limit_num_images: int = None,
                 other_class_adder: EnumClassPatchAdder = EnumClassPatchAdder.NoClassPatchAdder,
                 interval: int = 1):
        print("Using ClassificationCache")
        self.attr_standardization = True
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_limit_num_images = limit_num_images
        with File(f"{FolderInfos.input_data_folder}filtered_cache_annotations.hdf5", "r") as images_cache:
            self.tr_keys = list(images_cache.keys())[:int(len(images_cache) * tr_percent)]
            if self.attr_limit_num_images is not None:
                self.tr_keys = self.tr_keys[:self.attr_limit_num_images]
            self.valid_keys = list(images_cache.keys())[int(len(images_cache) * tr_percent):]
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

        if other_class_adder == EnumClassPatchAdder.OtherClassPatchAdder:
            self.attr_other_class_adder = OtherClassPatchAdder(interval=interval)
            self.attr_standardizer = StandardizerCacheMixed(interval=interval)
        elif other_class_adder == EnumClassPatchAdder.NoClassPatchAdder:
            self.attr_other_class_adder = NoClassPatchAdder(interval=interval)
            self.attr_standardizer = StandardizerCacheSeepSpill()
        self.attr_patch_adder_callback = PatchAdderCallback(class_adders=[self.attr_other_class_adder])
        with open(f"{FolderInfos.input_data_folder}filtered_img_infos.json", "r") as fp:
            self.dico_infos = json.load(fp)

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
        with File(f"{FolderInfos.input_data_folder}filtered_cache_annotations.hdf5", "r") as annotations_cache:
            with File(f"{FolderInfos.input_data_folder}filtered_cache_images.hdf5", "r") as images_cache:
                random.shuffle(images_available)
                self.attr_patch_adder_callback.on_epoch_start()
                for id in images_available:
                    data = self.attr_other_class_adder.generate_if_required()
                    if data is not None:
                        image, annotation, transformation_matrix, source_img = data
                        image, annotation = self.process_infos(image, annotation)
                        yield image, annotation, transformation_matrix, source_img
                    # Open image and annotations
                    image = np.array(images_cache[id])
                    annotation = np.array(annotations_cache[id])
                    source_img = self.dico_infos[id]["source_img"]
                    transformation_matrix = np.array(self.dico_infos[id]["transformation_matrix"])
                    image, annotation = self.process_infos(image, annotation)
                    yield image, annotation, transformation_matrix, source_img


    def process_infos(self, image, annotation):
        image = self.attr_standardizer.standardize(image)
        image = np.stack((image,) * 3, axis=0)
        annotation = self.attr_label_modifier.make_classification_label(annotation)
        return image, annotation

    def len(self, dataset) -> Optional[int]:
        if dataset == "tr":
            return len(self.tr_keys)
        else:
            return len(self.valid_keys)
