"""Annotations access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""

import pickle
import numpy as np
from PIL import Image,ImageDraw
from typing import List, Tuple

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.TwoWayDict import TwoWayDict
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class PointDataset(BaseClass,AbstractDataset):
    """Annotations access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""
    def __init__(self,path_pkl:str,mapping: TwoWayDict):
        super(PointDataset, self).__init__(mapping)
        self.attr_path = path_pkl
        with open(path_pkl,"rb") as fp:
            self._dataset = pickle.load(fp)

    @property
    def dataset(self):
        return self._dataset
    def get(self,item: str, transformation_matrix: np.ndarray, array_size: int) -> List[Tuple[int,int]]:
        """Gives access to the data and can simultenaously perform augmentations to constitute the final augmented annotation
        (give identity matrix for no transformation)

        Args:
            item: str, name of the sample in the annotation dataset
            transformation_matrix: transformation matrix to apply on the points before creating the final annotation array
            array_size: int, size of the output array

        Returns:
            segmentation_map, np.ndarray the segmentation map (the numpy array annotaiton)
        """
        polygons = []
        for i,polygon in enumerate(self.dataset[item]):
            label:int = self.attr_mapping[polygon[EnumShapeCategories.Label]]
            color_code = "#"+f"{label:02x}"*3
            polygons.append({EnumShapeCategories.Label:color_code,EnumShapeCategories.Points:polygon[EnumShapeCategories.Points]})
        return self.dataset[item]
        segmentation_map = np.zeros((array_size,array_size),dtype=np.uint8)
        segmentation_map = Image.fromarray(segmentation_map)
        draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
        for shape_dico in self.dico[item]:
            liste_points_shape = [tuple(transformation_matrix.dot([*point,1])[:2]) for point in shape_dico[EnumShapeCategories.Points]]
            label = shape_dico[EnumShapeCategories.Label]
            if label == "seep":  # Change color and so the value put in the array to create the label
                color = "#010101"
            elif label == "spill":
                color = "#020202"
            else:
                color = "#000000"
            draw.polygon(liste_points_shape, fill=color)
        segmentation_map = np.array(segmentation_map, dtype=np.uint8)
        return segmentation_map

    def __len__(self):
        return len(self.dataset)
    def values(self):
        return self.dataset.values()
    def keys(self):
        return self.dataset.keys()
    def __iter__(self):
        return self.dataset.__iter__()