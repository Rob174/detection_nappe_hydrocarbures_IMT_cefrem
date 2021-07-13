from h5py import File

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass


class NumpyAnnotations(BaseClass):
    def __init__(self):
        self.annotations_labels = File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5", "r")
    def __getitem__(self,item):
        return self.annotations_labels[item]
    def __len__(self):
        return len(self.annotations_labels)
    def values(self):
        return self.annotations_labels.values()
    def keys(self):
        return self.annotations_labels.keys()
    def __iter__(self):
        return self.annotations_labels.__iter__()