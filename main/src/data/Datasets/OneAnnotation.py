"""A dataset to manage hdf5 files for 2d np.ndarrays"""
import numpy as np
from h5py import File
from typing import Tuple, List

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class OneAnnotation(BaseClass, AbstractDataset):
    """A dataset to manage hdf5 files for 2d np.ndarrays"""

    def __init__(self, annotation: int, keys: List, shape:Tuple, mapping: TwoWayDict):
        super().__init__(mapping)
        self.attr_annotation = annotation
        self.attr_shape = shape
        self.keys_list = keys

    def get(self, id: str):
        """Get the object representing the array of id name

        Args:
            id: id of the sample in the dataset

        Returns:
            np.ndarray, representing the sample extracted from the dataset

        """
        return np.zeros(self.attr_shape,dtype=np.float32)+self.attr_annotation

    def __iter__(self):
        return (k for k in self.keys())

    def keys(self):
        return self.keys_list

    def values(self):
        return (self.get(id) for id in self.keys())

    def __len__(self):
        return len(self.keys())
