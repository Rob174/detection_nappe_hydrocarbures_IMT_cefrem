"""Class to get the a grid of patches in an image with affine transformations"""

import numpy as np
from typing import List, Tuple

from main.src.data.GridMaker.AbstractGridMaker import AbstractGridMaker
from main.src.param_savers.BaseClass import BaseClass


class GridMaker(BaseClass, AbstractGridMaker):
    """Class to get the a grid of patches in an image with affine transformations
    Args:
        patch_size_final_resize: int, final size of the patch before providing it to the model
    """

    def __init__(self, patch_size_final_resize: int):
        super(GridMaker, self).__init__(patch_size_final_resize)

    def get_grid(self, img_shape: Tuple[int, ...], partial_transformation_matrix: np.ndarray = None) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image with at the two first positions the with and height of the image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch

        NB: does not violates inherited method (Listov principle) as we have a default value for partial_transformation_matrix
        """
        if partial_transformation_matrix is None:
            partial_transformation_matrix = np.identity(3)
        rows, cols = img_shape[:2]
        original_mapped_corner1 = partial_transformation_matrix.dot([cols, rows, 1])
        original_mapped_corner2 = partial_transformation_matrix.dot([cols, rows, 1])
        max_rows = max(original_mapped_corner1[1], original_mapped_corner2[1]) - self.attr_patch_size_final_resize
        max_cols = max(original_mapped_corner1[0], original_mapped_corner2[0]) - self.attr_patch_size_final_resize
        cols_coords = np.arange(0, max_cols, self.attr_patch_size_final_resize)
        rows_coords = np.arange(0, max_rows, self.attr_patch_size_final_resize)
        coords = list(zip(*list(x.flat for x in np.meshgrid(rows_coords, cols_coords))))
        return coords

    def get_patch_transformation_matrix(self, coord_patch: Tuple[int, int]):
        """Get the transformation matrix corresponding to the coordinates provided

        Args:
            coord_patch: Tuple[int,int], x,y coordinates of the patch (as in numpy format)

        Returns:
            the np.ndarray of the transformation matrix to use to get the patch at the upper left corner of the initial image
        """
        return np.array([[1, 0, -coord_patch[1]],
                         [0, 1, -coord_patch[0]],
                         [0, 0, 1]])
