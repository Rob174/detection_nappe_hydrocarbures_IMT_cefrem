"""A dataset to manage hdf5 files for 2d np.ndarrays"""
import numpy as np

from h5py import File

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class ImageDataset(BaseClass,AbstractDataset):
    """A dataset to manage hdf5 files for 2d np.ndarrays"""
    def __init__(self, src_hdf5: str, mapping:TwoWayDict):
        super().__init__(mapping)
        self.attr_path = src_hdf5

    def get(self, id:str):
        """Get the object representing the array of id name

        Args:
            id: id of the sample in the dataset

        Returns:
            np.ndarray, representing the sample extracted from the dataset

        """
        with File(self.attr_path,"r") as file:
            return np.array(file,dtype=np.float32)
    def __iter__(self):
        """Allow to use for loop on this object"""
        raise NotImplementedError
    def keys(self):
        with File(self.attr_path,"r") as file:
            return list(file.keys())
    def values(self):
        with File(self.attr_path,"r") as file:
            return list(file.values())

    def __len__(self):
        with File(self.attr_path,"r") as dataset:
            return len(dataset)