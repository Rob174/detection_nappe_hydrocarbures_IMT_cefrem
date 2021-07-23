from h5py import File

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class ImageDataset(BaseClass,AbstractDataset):
    def __init__(self, src_hdf5: str, mapping:TwoWayDict):
        super().__init__(mapping)
        self.attr_path = src_hdf5
        self._dataset = File(src_hdf5,"r")
    @property
    def dataset(self):
        return self._dataset
    def get(self,id:str):
        return self.dataset[id]
    def keys(self):
        return self.dataset.keys()
    def values(self):
        return self.dataset.values()

    def __enter__(self):
        self.dataset.__enter__()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dataset.__exit__(exc_type,exc_val,exc_tb)


