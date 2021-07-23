import json
from functools import lru_cache
from typing import Tuple

import numpy as np
from h5py import File

import main.src.data.resizer as resizer
from main.FolderInfos import FolderInfos
from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class DataSentinel1Segmentation(BaseClass):
    attr_original_class_mapping = TwoWayDict(  # a twoway dict allowing to store pairs of hashable objects:
        {  # Formatted in the following way: src_index in cache, name, the position encode destination index
            0: "other",
            1: "seep",
            2: "spill",
        })

    def __init__(self, limit_num_images=None, input_size=None):
        """Class giving access and managing the original attr_dataset stored in the hdf5 and json files

        Args:
            limit_num_images: limit the number of image in the attr_dataset per epoch (before filtering)
            input_size: the size of the image provided as input to the attr_model ⚠️
        """
        self.attr_with_normalization = True
        self.attr_name = self.__class__.__name__
        # Opening the cache
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            self.images_infos = json.load(fp)
        with open(f"{FolderInfos.input_data_folder}pixel_stats.json", "r") as fp:
            self.pixel_stats = json.load(fp)
        self.images = File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r")
        self.annotations_labels = None
        self.attr_limit_num_images = limit_num_images
        # concretely we can ask:
        # - self.attr_class_mapping[0] -> returns "other" as a normal dict
        # - self.attr_class_mapping["other"] -> returns 0 with this new type of object
        self.attr_class_mapping = TwoWayDict(
            {k: v for k, v in DataSentinel1Segmentation.attr_original_class_mapping.items()})
        self.attr_resizer = resizer.Resizer(
            out_size_w=input_size)  # resize object used to resize the image to the final size for the network



    def close(self):
        """
        Close hdf5 objects properly (not mandatory for read from what i have seen)
        Returns:

        """
        self.images.close()
        self.annotations_labels.close()

