"""Fabrics access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""

import pickle
from typing import List, Dict, Union

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.ColorExtractor.AbstractColorExtractor import AbstractColorExtractor
from main.src.data.Datasets.PointExtractor.AbstractPointExtractor import AbstractPointExtractor
from main.src.data.TwoWayDict import TwoWayDict
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class PointDataset(BaseClass, AbstractDataset):
    """Fabrics access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""

    def __init__(self, path_pkl: str, mapping: TwoWayDict, color_extractor: AbstractColorExtractor,point_extractor: AbstractPointExtractor):
        super(PointDataset, self).__init__(mapping)
        self.attr_path = path_pkl
        with open(path_pkl, "rb") as fp:
            self._dataset = pickle.load(fp)
        self.attr_color_extractor = color_extractor
        self.attr_point_extractor = point_extractor

    @property
    def dataset(self):
        return self._dataset

    def get(self, item: str) -> List[Dict[EnumShapeCategories, Union[str, List]]]:
        """Gives access to the data and can simultenaously perform augmentations to constitute the final augmented annotation
        (give identity matrix for no transformation)

        Args:
            item: str, name of the sample in the annotation dataset

        Returns:
            segmentation_map, np.ndarray the segmentation map (the numpy array annotaiton)
        """
        polygons = []
        for i, polygon in enumerate(self.dataset[item]):
            color_code = self.attr_color_extractor.extract(polygon)
            points = self.attr_point_extractor.extract(polygon)
            polygons.append({EnumShapeCategories.Label: color_code,
                             EnumShapeCategories.Points: points})
        return polygons
    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self.dataset.__iter__()
