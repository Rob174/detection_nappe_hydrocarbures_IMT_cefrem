import cv2
import numpy as np
from typing import Tuple

from main.src.param_savers.BaseClass import BaseClass


class AugmentationApplierLabelPoints(BaseClass):
    def __init__(self, access_function,grid_maker,patch_size_final_resize: int):
        self.access_function = access_function
        self.grid_maker = grid_maker
        self.patch_size_final_resize = patch_size_final_resize
    def transform(self,image_name: str, partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]):

        transformation_matrix = self.grid_maker.get_patch_transformation_matrix(patch_upper_left_corner_coords).dot(
                                partial_transformation_matrix)
        label = self.access_function(image_name, transformation_matrix, self.patch_size_final_resize)
        return label,transformation_matrix
