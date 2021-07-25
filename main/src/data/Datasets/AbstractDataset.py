"""Base class to build your own dataset"""
from abc import ABC
from typing import Any

from main.src.data.TwoWayDict import TwoWayDict


class AbstractDataset(ABC):
    """Base class to build your own dataset"""
    def __init__(self,mapping: TwoWayDict,*args,**kwargs):
        self.attr_mapping = mapping

    @property
    def dataset(self):
        """property to map to the dataset object mappping to the file (hdf5 object, ...)"""
        raise NotImplementedError

    def __getitem__(self,id: str) -> Any:
        """Get the object representing the array of id name

        Args:
            id: id of the sample in the dataset

        Returns:
            Any, object representing the sample extracted from the dataset

        """
    def __len__(self):
        return self.dataset
    def keys(self):
        return self.dataset.keys()
    def values(self):
        return self.dataset.values()
    def __iter__(self):
        """Allow to use for loop on this object"""
        return self.dataset.__iter__()
    def __enter__(self):
        """Method for context manager (with ... statement)"""
        self.dataset_opened = self.dataset.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Method for context manager (with ... statement)"""
        self.dataset.__exit__(exc_type,exc_val,exc_tb)