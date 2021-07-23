from abc import ABC
from typing import Any

from main.src.data.TwoWayDict import TwoWayDict


class AbstractDataset(ABC):
    """Object representing a dataset"""
    def __init__(self,mapping: TwoWayDict,*args,**kwargs):
        self.attr_mapping = mapping

    @property
    def dataset(self):
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
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass