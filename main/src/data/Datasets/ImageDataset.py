"""A dataset to manage hdf5 files for 2d np.ndarrays"""
from h5py import File

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class ImageDataset(BaseClass,AbstractDataset):
    """A dataset to manage hdf5 files for 2d np.ndarrays"""
    def __init__(self, src_hdf5: str, mapping:TwoWayDict):
        super().__init__(mapping)
        self.attr_path = src_hdf5
        self._dataset = File(src_hdf5,"r")

    @property
    def dataset(self):
        """mapping to the HDF5 object file"""
        return self._dataset
    def __getitem__(self, id:str):
        """Get the object representing the array of id name

        Args:
            id: id of the sample in the dataset

        Returns:
            np.ndarray, representing the sample extracted from the dataset

        """
        raise Exception("Directly managed by the hdf5 file")
    def __enter__(self,*args,**kwargs):
        return self.dataset.__enter__(*args,**kwargs)
    def __exit__(self, *args, **kwargs):
        return self.dataset.__exit__( *args, **kwargs)
    def __iter__(self):
        """Allow to use for loop on this object"""
        raise NotImplementedError
    def __len__(self):
        return len(self.dataset)