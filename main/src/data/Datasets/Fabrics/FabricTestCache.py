
import json
from typing import Tuple, Dict

from main.FolderInfos import FolderInfos as FI
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.HDF5Dataset import HDF5Dataset
from main.src.data.Datasets.OneAnnotation import OneAnnotation
from main.src.data.TwoWayDict import TwoWayDict

class FabricTestCache:
    """Creates the group of datasets representing the filtred cache with
    - filtered_cache_images.hdf5
    - filtered_cache_annotations.hdf5
    - filtered_img_infos.json
    """

    def __call__(self) -> Tuple[AbstractDataset, AbstractDataset, Dict]:
        """Creates the object representing this dataset"""
        mapping = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                1: "seep",
                2: "spill",
            })
        images = HDF5Dataset(
            f"{FI.input_data_folder}test_cache{FI.separator}test_cache_images.hdf5",
            mapping=mapping)
        annotations = HDF5Dataset(
            f"{FI.input_data_folder}test_cache{FI.separator}test_cache_annotations.hdf5",
            mapping=mapping)
        with open(
                f"{FI.input_data_folder}test_cache{FI.separator}test_cache_img_infos.json",
                "r") as fp:
            informations = json.load(fp)
        return images, annotations, informations
