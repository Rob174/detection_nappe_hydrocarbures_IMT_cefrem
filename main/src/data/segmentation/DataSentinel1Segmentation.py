import json
from functools import lru_cache
from typing import Tuple

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
import main.src.data.resizer as resizer
import copy

from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class DataSentinel1Segmentation(BaseClass):
    attr_original_class_mapping = TwoWayDict(  # a twoway dict allowing to store pairs of hashable objects:
    {  # Formatted in the following way: src_index in cache, name, the position encode destination index
        0: "other",
        1: "spill",
        2: "seep",
    })

    def __init__(self,limit_num_images=None,input_size=None):
        """Class giving access and managing the original dataset stored in the hdf5 and json files

        Args:
            limit_num_images: limit the number of image in the dataset per epoch (before filtering)
            input_size: the size of the image provided as input to the model ⚠️
        """
        self.attr_with_normalization = True
        self.attr_name = self.__class__.__name__
        # Opening the cache
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json","r") as fp:
            self.images_infos = json.load(fp)
        with open(f"{FolderInfos.input_data_folder}pixel_stats.json","r") as fp:
            self.pixel_stats = json.load(fp)
        self.images = File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r")
        self.annotations_labels = File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5", "r")
        self.attr_limit_num_images = limit_num_images
        #concretely we can ask:
        # - self.attr_class_mapping[0] -> returns "other" as a normal dict
        # - self.attr_class_mapping["other"] -> returns 0 with this new type of object
        self.attr_class_mapping = TwoWayDict({k:v for k,v in DataSentinel1Segmentation.attr_original_class_mapping.items()})
        self.attr_resizer = resizer.Resizer(out_size_w=input_size) # resize object used to resize the image to the final size for the network

        self.attr_global_name = "dataset"
    def __get__(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.getitem(id)
    def getimage(self,name: str) -> np.ndarray:
        img = np.array(self.images[name])
        img = (img-self.pixel_stats["mean"])/self.pixel_stats["std"]
        return img
    def getitem(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Magic python method to get the item of global id asked

        Args:
            id: int, global id of the sample

        Returns:
            - image, np.ndarray 2d image resized to the size specified in constructor
            - annotation, np.ndarray 2d annotation resized to the size specified in constructor
        """
        # Get the image uniq id corresponding to this global id

        item = self.get_all_items()[id]
        img = self.getimage(item)
        self.current_name = item # for tests if we need access to the last image processed
        return self.attr_resizer(img), self.attr_resizer(self.annotations_labels[item])
    @lru_cache(maxsize=1)
    def get_all_items(self):
        """List available original images available in the dataset (hdf5 file)
        the :lru_cache(maxsize=1) allow to compute it only one time and store the result in a cache

        Allow to limit the number of original image used in the dataset

        Returns:
            list of str: [img_uniq_id0,...]

        """
        if self.attr_limit_num_images is not None:
            return list(self.images.keys())[:self.attr_limit_num_images]
        return list(self.images.keys())

    def close(self):
        """
        Close hdf5 objects properly (not mandatory for read from what i have seen)
        Returns:

        """
        self.images.close()
        self.annotations_labels.close()

    def __len__(self) -> int:
        """Magic method called when we make len(obj)"""
        return len(self.get_all_items())

# Usage: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html https://pytorch.org/docs/stable/data.html
# data = DataLoader(DataSentinel1Segmentation(), batch_size=1, shuffle=False, sampler=None,
#                   batch_sampler=None, num_workers=0, collate_fn=None,
#                   pin_memory=False, drop_last=False, timeout=0,
#                   worker_init_fn=None, prefetch_factor=2,
#                   persistent_workers=False)

# if __name__ == "__main__":
