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
        """mapping to the HDF5 object file"""
        return self._dataset
    def get(self,id:str):
        """Get the object representing the array of id name

        Args:
            id: id of the sample in the dataset

        Returns:
            np.ndarray, representing the sample extracted from the dataset

        """
        return self.dataset[id]