import json

from main.src.data.Datasets.DatasetOpener.AbstractOpener import AbstractOpener
from main.src.data.Datasets.DatasetOpener.HDF5Opener import HDF5Opener
from main.src.data.Datasets.DatasetOpener.JSONOpener import JSONOpener
from main.src.data.Datasets.HDF5Dataset import HDF5Dataset


class OptimizedMemoryPointOpener(AbstractOpener):
    def __init__(self, path_hdf5: str,path_json: str):
        super(OptimizedMemoryPointOpener, self).__init__("")

        self.attr_json = JSONOpener(path_json)
        self.attr_hdf5 = HDF5Opener(path_hdf5)
    def __getitem__(self, item):
        lengthes_of_polygons = self.attr_json[item]
        polygons = [[]]
        current_polygon_id = 0
        for point in self.attr_hdf5[item]:
            if len(polygons[-1]) == lengthes_of_polygons[current_polygon_id]:
                current_polygon_id += 1
                polygons.append([])
            polygons[-1].append(point)
        return polygons
