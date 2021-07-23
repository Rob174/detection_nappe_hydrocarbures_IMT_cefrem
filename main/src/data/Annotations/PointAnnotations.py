"""Annotations access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""

import pickle
import numpy as np
from PIL import Image,ImageDraw

from main.FolderInfos import FolderInfos
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class PointAnnotations(BaseClass):
    """Annotations access thanks to the images_preprocessed_points.pkl file containing points of annotations polygons"""
    def __init__(self):
        with open(FolderInfos.input_data_folder+"images_preprocessed_points.pkl","rb") as fp:
            self.dico = pickle.load(fp)

    def get(self,item: str, transformation_matrix: np.ndarray, array_size: int) -> np.ndarray:
        """Gives access to the data and can simultenaously perform augmentations to constitute the final augmented annotation
        (give identity matrix for no transformation)

        Args:
            item: str, name of the sample in the annotation dataset
            transformation_matrix: transformation matrix to apply on the points before creating the final annotation array
            array_size: int, size of the output array

        Returns:
            segmentation_map, np.ndarray the segmentation map (the numpy array annotaiton)
        """
        assert transformation_matrix.shape == (3,3), f"Invalid shape {transformation_matrix.shape} for transformation_matrix"
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
        return len(self.dico)
    def values(self):
        return self.dico.values()
    def keys(self):
        return self.dico.keys()
    def __iter__(self):
        return self.dico.__iter__()