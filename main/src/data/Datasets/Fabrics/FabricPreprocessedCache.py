"""Creates the group of datasets representing the filtred cache with

    - images_preprocessed.hdf5
    - images_preprocessed_points.pkl
    - images_informations_preprocessed.json
    """

import json
from typing import Tuple, Dict

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.ImageDataset import ImageDataset
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.TwoWayDict import TwoWayDict


class FabricPreprocessedCache:
    """Creates the group of datasets representing the filtred cache with
        - images_preprocessed.hdf5
        - images_preprocessed_points.pkl
        - images_informations_preprocessed.json
        """
    def __call__(self) -> Tuple[AbstractDataset,AbstractDataset,Dict]:
        """Creates the object representing this dataset"""
        mapping = TwoWayDict({  # Formatted in the following way: src_index in cache, name, the position encode destination index
                                0: "other",
                                1: "seep",
                                2: "spill",
                            })
        images = ImageDataset(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5",mapping=mapping)
        annotations = PointDataset(FolderInfos.input_data_folder+"images_preprocessed_points.pkl",mapping=mapping)
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            informations = json.load(fp)
        return images,annotations,informations