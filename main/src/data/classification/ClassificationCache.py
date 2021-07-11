import json
import random
from typing import Tuple, Generator, Optional

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.TwoWayDict import Way
from main.src.data.classification.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.classification.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.classification.LabelModifier.NoLabelModifier import NoLabelModifier
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
from main.src.param_savers.BaseClass import BaseClass


class ClassificationCache(BaseClass):
    """Class that takes inputs from the filtered hdf5 file and
    filter them.

    Args:

        label_modifier: EnumLabelModifier
    """

    def __init__(self, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent: float = 0.7, limit_num_images: int = None):
        print("Using ClassificationCache")
        self.attr_standardization = False
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_limit_num_images = limit_num_images
        with File(f"{FolderInfos.input_data_folder}filtered_cache_annotations.hdf5", "r") as images_cache:
            self.tr_keys = list(images_cache.keys())[:int(len(images_cache) * tr_percent)]
            if self.attr_limit_num_images is not None:
                self.tr_keys = self.tr_keys[:self.attr_limit_num_images]
            self.valid_keys = list(images_cache.keys())[int(len(images_cache) * tr_percent):]
        self.attr_global_name = "attr_dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = NoLabelModifier()
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")

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
                for id in images_available:
                    image = np.array(images_cache[id])
                    image = np.stack((image,) * 3, axis=0)
                    annotation = np.array(annotations_cache[id])
                    annotation = self.make_classification_label(annotation)
                    annotation = self.attr_label_modifier.make_classification_label(annotation)
                    source_img = self.dico_infos[id]["source_img"]
                    transformation_matrix = np.array(self.dico_infos[id]["transformation_matrix"])
                    yield image, annotation, transformation_matrix, source_img

    def make_classification_label(self, annotations_patch):
        """Creates the classification label based on the annotation patch image

        Indicates if we need to reject the patch due to overrepresented class

        Args:
            annotations_patch: np.ndarray 2d containing for each pixel the class of this pixel

        Returns: the classification label

        """

        output = np.zeros((len(DataSentinel1Segmentation.attr_original_class_mapping),), dtype=np.float32)
        for value in DataSentinel1Segmentation.attr_original_class_mapping.keys(Way.ORIGINAL_WAY):
            # for each class of the original attr_dataset, we put a probability of presence of one if the class is in the patch
            value = int(value)
            #  if the class is in the patch
            if value in annotations_patch:
                output[value] = 1.
        return output

    def len(self, dataset) -> Optional[int]:
        if dataset == "tr":
            return len(self.tr_keys)
        else:
            return len(self.valid_keys)
