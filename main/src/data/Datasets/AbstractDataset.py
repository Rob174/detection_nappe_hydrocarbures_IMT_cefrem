from abc import ABC
from typing import Any

from main.src.data.TwoWayDict import TwoWayDict


class AbstractDataset(ABC):
    """Object representing a dataset"""
    def __init__(self,mapping: TwoWayDict,*args,**kwargs):
        self.attr_mapping = mapping

    @property
    def dataset(self):
        """property to map to the dataset object mappping to the file (hdf5 object, ...)"""
        raise NotImplementedError

    def get(self,id: str) -> Any:
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
    def __enter__(self):
        """Method for context manager (with ... statement)"""
        self.dataset.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Method for context manager (with ... statement)"""
        self.dataset.__exit__(exc_type,exc_val,exc_tb)