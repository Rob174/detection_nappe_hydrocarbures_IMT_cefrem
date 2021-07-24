"""To apply a transformation matrix on a list of points and generate the corresponding patch"""

from typing import Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw

from main.src.data.Augmentation.Augmentations.AugmentationApplier.AbstractApplier import AbstractApplier
from main.src.data.GridMaker.GridMaker import GridMaker
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories
from main.src.param_savers.BaseClass import BaseClass


class AugmentationApplierLabelPoints(BaseClass, AbstractApplier):
    """To apply a transformation matrix on a list of points and generate the corresponding patch

    Args:
        grid_maker: GridMaker, allowing to get the transformation matrix corresponding to the an extraction of a patch at specified coordinates
        patch_size_final_resize: int, final size of the patch to be supplied to the model
    """

    def __init__(self, grid_maker: GridMaker, patch_size_final_resize: int):
        super(AugmentationApplierLabelPoints, self).__init__(grid_maker, patch_size_final_resize)

    def transform(self, data: List[Dict], partial_transformation_matrix: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Draw the polygons transformed thanks to the transformation matrix to build the annotation of the patch

        Args:
            data: List[Dict] containing the "points" and the "label" keys containing respectively a list of points defining the polygon and the color hexadecimal code to use
            partial_transformation_matrix: np.ndarray, transformation to apply without the translation to get the correct patch
            patch_upper_left_corner_coords: Tuple[int,int], coordinates of the upper left corner of the patch to take

        Returns:
            (image,transformation_matrix), tuple:
                - segmentation_map: np.ndarray: resulting patch
                - transformation_matrix: np.ndarray, final transformation matrix applied
        """
        transformation_matrix = self.grid_maker.get_patch_transformation_matrix(patch_upper_left_corner_coords).dot(
            partial_transformation_matrix)
        segmentation_map = np.zeros((self.patch_size_final_resize, self.patch_size_final_resize), dtype=np.uint8)
        segmentation_map = Image.fromarray(segmentation_map)
        draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
        for shape_dico in data:
            liste_points_shape = [tuple(transformation_matrix.dot([*point, 1])[:2]) for point in
                                  shape_dico[EnumShapeCategories.Points]]
            color = shape_dico[EnumShapeCategories.Label]
            draw.polygon(liste_points_shape, fill=color)
        segmentation_map = np.array(segmentation_map, dtype=np.uint8)
        return segmentation_map, transformation_matrix
