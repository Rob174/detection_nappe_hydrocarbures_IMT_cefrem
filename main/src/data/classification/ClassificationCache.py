import json
import random

import numpy as np
from typing import Tuple, Generator
from h5py import File

from main.FolderInfos import FolderInfos
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
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep,EnumClasses.Spill),
                 tr_percent: float = 0.7, limit_num_images: int = None):
        self.attr_name = self.__class__.__name__ # save the name of the class used for reproductibility purposes
        self.attr_limit_num_images = limit_num_images
        with File(f"{FolderInfos.input_data_folder}filtered_cache_annotations.hdf5","r") as images_cache:
            self.tr_keys = list(images_cache.keys())[:int(len(images_cache)*tr_percent)]
            self.valid_keys = list(images_cache.keys())[int(len(images_cache)*tr_percent):]
        self.attr_global_name = "dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = NoLabelModifier()
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,original_class_mapping=DataSentinel1Segmentation.attr_class_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")

        with open(f"{FolderInfos.input_data_folder}filtered_img_infos.json", "r") as fp:
            self.dico_infos = json.load(fp)


    def __iter__(self, dataset="tr"):
        return iter(self.generator())
    def generator(self,dataset="tr") -> Generator[Tuple[np.ndarray,np.ndarray,np.ndarray,str],None,None]:
        """Generator that generates data for the trainer

        Args:
            dataset: str, tr or valid to choose source images for tr or valid dataset

        Returns:
            generator of the dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   transformation_matrix:  the transformation matrix to transform the source image
                   item: str name of the source image
        """
        images_available = self.tr_keys if dataset=="tr" else self.valid_keys
        if self.attr_limit_num_images is not None and dataset=="tr":
            images_available = images_available[:self.attr_limit_num_images]
        with File(f"{FolderInfos.input_data_folder}filtered_cache_annotations.hdf5","r") as images_cache:
            with File(f"{FolderInfos.input_data_folder}filtered_cache_images.hdf5", "r") as annotations_cache:
                random.shuffle(images_available)
                for id in images_available:
                    image = np.array(images_cache[id])
                    annotation = np.array(annotations_cache[id])
                    annotation = self.attr_label_modifier.make_classification_label(annotation)
                    source_img = self.dico_infos[id]["source_img"]
                    transformation_matrix = np.array(self.dico_infos[id]["transformation_matrix"])
                    yield image, annotation, transformation_matrix, source_img