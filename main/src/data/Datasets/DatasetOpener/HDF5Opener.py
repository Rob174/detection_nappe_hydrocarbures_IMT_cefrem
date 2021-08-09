import json
from h5py import File
from main.src.data.Datasets.DatasetOpener.AbstractOpener import AbstractOpener


class HDF5Opener(AbstractOpener):
    def __init__(self, path: str):
        super(HDF5Opener, self).__init__(path)
        self.dataset = File(self.attr_path, "r")