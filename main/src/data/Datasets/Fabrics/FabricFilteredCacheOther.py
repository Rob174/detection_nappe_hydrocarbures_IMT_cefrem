"""Creates the group of datasets representing the filtred cache with

    - filtered_cache_images.hdf5
    - filtered_cache_annotations.hdf5
    - filtered_img_infos.json
    """
import json
from typing import Tuple, Dict

from main.FolderInfos import FolderInfos as FI, FolderInfos
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.HDF5Dataset import HDF5Dataset
from main.src.data.Datasets.OneAnnotation import OneAnnotation
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.TwoWayDict import TwoWayDict, Way


class FabricFilteredCacheOther:
    """Creates the group of datasets representing the filtred cache with
    - filtered_cache_images.hdf5
    - filtered_cache_annotations.hdf5
    - filtered_img_infos.json
    """

    def __call__(self) -> Tuple[AbstractDataset, AbstractDataset, Dict]:
        """Creates the object representing this dataset"""
        seep_spill_mapping = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                1: "seep",
                2: "spill"
            })
        global_mapping = TwoWayDict(
            {
                **seep_spill_mapping.attr_dico_one_way,
                3:"earth"
            })
        images = HDF5Dataset(f"{FI.input_data_folder}filtered_cache_other{FI.separator}filtered_cache_other_images.hdf5",
                             mapping=global_mapping)
        annotations = PointDataset( # TODO : transform preprocessed earth annotations into filtered annotations
            path_pkl=FolderInfos.input_data_folder+
        )

        with open(f"{FI.input_data_folder}filtered_cache_other{FI.separator}filtered_cache_other_img_infos.json", "r") as fp:
            informations = json.load(fp)
        return images, annotations, informations
