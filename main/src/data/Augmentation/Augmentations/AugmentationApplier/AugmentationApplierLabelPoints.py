from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, Callable, List, Dict

from main.src.data.GridMaker.GridMaker import GridMaker
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class AugmentationApplierLabelPoints(BaseClass):
    def __init__(self, access_function: Callable[[str,np.ndarray,int],np.ndarray],grid_maker: GridMaker,patch_size_final_resize: int):
        """

        Args:
            access_function: Callable[[str,np.ndarray,int],np.ndarray] allowing to retrive an annotation and automatically apply the transformation matrix to get the resulting np.ndarray
            grid_maker: GridMaker, allowing to get the transformation matrix corresponding to the an extraction of a patch at specified coordinates
            patch_size_final_resize: int, final size of the patch to be supplied to the model
        """
        self.access_function = access_function
        self.grid_maker = grid_maker
        self.patch_size_final_resize = patch_size_final_resize
    def transform(self,polygons: List[Dict], partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray,np.ndarray]:
        """Apply the transformation on the points thanks to the PointAnnotations get object method

        Args:
            image: np.ndarray
            partial_transformation_matrix: np.ndarray, transformation to apply without the translation to get the correct patch
            patch_upper_left_corner_coords: Tuple[int,int], coordinates of the upper left corner of the patch to take

        Returns:
            (image,transformation_matrix), tuple:
                - image: np.ndarray: resulting patch
                - transformation_matrix: np.ndarray, final transformation matrix applied
        """
        transformation_matrix = self.grid_maker.get_patch_transformation_matrix(patch_upper_left_corner_coords).dot(
                                partial_transformation_matrix)
        segmentation_map = np.zeros((self.patch_size_final_resize, self.patch_size_final_resize), dtype=np.uint8)
        segmentation_map = Image.fromarray(segmentation_map)
        draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
        for shape_dico in polygons:
            liste_points_shape = [tuple(transformation_matrix.dot([*point, 1])[:2]) for point in
                                  shape_dico[EnumShapeCategories.Points]]
            color = shape_dico[EnumShapeCategories.Label]
            draw.polygon(liste_points_shape, fill=color)
        segmentation_map = np.array(segmentation_map, dtype=np.uint8)
        return segmentation_map,transformation_matrix
