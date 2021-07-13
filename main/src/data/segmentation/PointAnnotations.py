import pickle
import numpy as np
from PIL.Image import Image
from PIL import ImageDraw

from main.FolderInfos import FolderInfos
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class PointAnnotations(BaseClass):
    def __init__(self):
        with open(FolderInfos.input_data_folder+"images_preprocessed_points.pkl","rb") as fp:
            self.dico = pickle.load(fp)

    def __getitem__(self,item: str,transformation_matrix: np.ndarray,array_size: int = 256):
        assert transformation_matrix.shape == (3,3), f"Invalid shape {transformation_matrix.shape} for transformation_matrix"
        transformed_coords = []
        for shape_dico in self.dico[item]:
            points = [transformation_matrix.dot([*shape_dico[EnumShapeCategories.Points],1])[:2]]
            transformed_coords.append({
                EnumShapeCategories.Label: shape_dico[EnumShapeCategories.Label],
                EnumShapeCategories.Points: points,
            })
    def draw(self,points_shapes,array):
        segmentation_map = Image.fromarray(array)
        draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
        for dico in points_shapes:
            label = dico[EnumShapeCategories.Label]
            liste_points_shape = dico[EnumShapeCategories.Points]
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