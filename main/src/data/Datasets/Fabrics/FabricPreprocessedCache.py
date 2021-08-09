"""Creates the group of datasets representing the filtred cache with

    - images_preprocessed.hdf5
    - images_preprocessed_points.pkl
    - images_informations_preprocessed.json
    """

import json
from typing import Tuple, Dict

from main.FolderInfos import FolderInfos as FI, FolderInfos
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.ColorExtractor.DicoColorExtractor import DicoColorExtractor
from main.src.data.Datasets.ColorExtractor.LayerColorExtractor import LayerColorExtractor
from main.src.data.Datasets.DatasetOpener.JSONOpener import JSONOpener
from main.src.data.Datasets.DatasetOpener.PickleOpener import PickleOpener
from main.src.data.Datasets.ImageDataset import ImageDataset
from main.src.data.Datasets.MergedImageDataset import MergedImageDataset
from main.src.data.Datasets.MergedPointDataset import MergedPointDataset
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.Datasets.PointExtractor.PointExtractorSingleCategoryDict import PointExtractorSingleCategoryDict
from main.src.data.Datasets.PointExtractor.PointExtractorMultiCategoryDict import PointExtractorMultiCategoryDict
from main.src.data.TwoWayDict import TwoWayDict


class FabricPreprocessedCache:
    """Creates the group of datasets representing the filtred cache with
        - images_preprocessed.hdf5
        - images_preprocessed_points.pkl
        - images_informations_preprocessed.json
        """

    def __call__(self) -> Tuple[AbstractDataset, AbstractDataset, Dict]:
        """Creates the object representing this dataset"""
        mapping = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                1: "seep",
                2: "spill",
            })
        images = ImageDataset(
            f"{FI.input_data_folder + FI.separator}preprocessed_cache{FI.separator}images_preprocessed.hdf5",
            mapping=mapping)
        annotations = PointDataset(
            color_extractor=DicoColorExtractor(mapping),
            point_extractor=PointExtractorMultiCategoryDict(),
            opener=PickleOpener(FI.input_data_folder + FI.separator + "preprocessed_cache" + FI.separator + "images_preprocessed_points.pkl")
        )
        mapping_earth = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                3: "earth"
            })
        annotations_earth = PointDataset(
            color_extractor=LayerColorExtractor(mapping_earth,fixed_color_code="#030303"),
            point_extractor=PointExtractorSingleCategoryDict(),
            opener=JSONOpener(FI.input_data_folder + FI.separator + "preprocessed_cache" + FI.separator + "world_earth_data.json")
        )
        annotations = MergedPointDataset(annotations, annotations_earth)
        informations = JSONOpener(f"{FI.input_data_folder}preprocessed_cache{FI.separator}images_informations_preprocessed.json").dataset
        return images, annotations, informations
