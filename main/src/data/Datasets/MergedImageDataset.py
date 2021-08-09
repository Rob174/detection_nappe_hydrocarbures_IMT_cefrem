import numpy as np
import typing

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.param_savers.BaseClass import BaseClass

class MergedImageDataset(AbstractDataset):

    def __init__(self, *datasets: AbstractDataset):
        self.attr_datasets: typing.Tuple[AbstractDataset] = datasets
        mapping = TwoWayDict({m["name"]:m["id"] for d in datasets for m in d.attr_mapping})
        super(MergedImageDataset, self).__init__(mapping)

    def get(self,id:str):
        # Roughly a reduce without going out of python world (not the case with functools)
        it = iter(self.attr_datasets)
        prec_data = next(it).get(id)

        for dataset in it:
            data = dataset.get(id)
            prec_data = np.max(np.stack([prec_data,data],axis=0),axis=0)
        return prec_data

    def keys(self):
        return self.attr_datasets[0].keys()

    def values(self):
        return (self.get(id) for id in self.keys())