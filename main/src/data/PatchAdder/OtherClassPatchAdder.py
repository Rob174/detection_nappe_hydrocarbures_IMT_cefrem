"""Class that allows to generate controlled number of patches everry ... number of annotated patch with only the other class"""

import json
import random
from typing import Optional, Tuple

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.PatchAdder.AbstractClassAdder import AbstractClassAdder
from main.src.param_savers.BaseClass import BaseClass


class OtherClassPatchAdder(BaseClass, AbstractClassAdder):
    """Generate determined number of patches with only the other class

    Args:
        interval: int, interval between two un annotated patches
    """

    def __init__(self, interval):
        super(OtherClassPatchAdder, self).__init__(interval)
        self.cache_img = File(FolderInfos.input_data_folder + "filtered_cache_other"+FolderInfos.separator+"filtered_cache_other_images.hdf5", "r")
        with open(FolderInfos.input_data_folder + "filtered_cache_other"+FolderInfos.separator+"filtered_cache_other_img_infos.json", "r") as fp:
            self.cache_infos = json.load(fp)
        self.num_annotated_btwn = 0

        self.ordered_keys = list(set(self.cache_img.keys()).intersection(set(self.cache_infos.keys())))
        self.keys_iterator = None

    def reinitialize_iter(self):
        """Method to restart from the beginning of the dataset the generation of ids"""
        random.shuffle(self.ordered_keys)
        self.keys_iterator = iter(self.ordered_keys)

    def next_index(self):
        """Generates the next index in the source dataset to use"""
        try:
            id = next(self.keys_iterator)
        except (StopIteration, TypeError):
            self.reinitialize_iter()
            id = next(self.keys_iterator)
        return id

    def generate_if_required(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """Method that generates a sample if it is the turn of the patch adder based on the interval to wait provided in the constructor

        Returns:
            Optional[Tuple]
            - if it is this dataset turn: patch_image, patch_annotation, transformation_matrix (used to build the patches), source image name
            - else None
        """
        if self.num_annotated_btwn == self.attr_interval:
            self.num_annotated_btwn = 0
            id = self.next_index()
            patch = np.array(self.cache_img[id], dtype=np.float32)
            annotation = np.zeros(patch.shape, dtype=np.float32)
            transformation_matrix = np.array(self.cache_infos[id]["transformation_matrix"])
            return patch, annotation, transformation_matrix, self.cache_infos[id]["source_img"]
        else:
            self.num_annotated_btwn += 1
            return None
