import numpy as np
import typing

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.param_savers.BaseClass import BaseClass

class MergedPointDataset(AbstractDataset):

    def __init__(self, *datasets: AbstractDataset):
        mapping = TwoWayDict({k:v for d in datasets for k,v in d.attr_mapping.items(Way.ORIGINAL_WAY)})
        super(MergedPointDataset, self).__init__(mapping)
        self.attr_datasets: typing.Tuple[AbstractDataset] = datasets

    def get(self,id:str):
        # Roughly a reduce without going out of python world (not the case with functools)
        it = iter(self.attr_datasets)
        polygons = []
        polygons.extend(next(it).get(id))

        for dataset in it:
            polygons.extend(dataset.get(id))
        return polygons

    def keys(self):
        return self.attr_datasets[0].keys()

    def values(self):
        return (self.get(id) for id in self.keys())
    def __len__(self):
        return len(self.keys())