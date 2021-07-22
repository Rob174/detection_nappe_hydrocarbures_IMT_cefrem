import cv2
import numpy as np
from typing import Tuple, Callable, Optional

from main.src.param_savers.BaseClass import BaseClass


class AugmentationApplierImage(BaseClass):
    def __init__(self,grid_maker,patch_size_final_resize: int):
        self.grid_maker = grid_maker
        self.patch_size_final_resize = patch_size_final_resize
    def transform(self,image: np.ndarray, partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]):
        transformation_matrix = self.grid_maker.get_patch_transformation_matrix(
            patch_upper_left_corner_coords).dot(partial_transformation_matrix)
        image = cv2.warpAffine(image, transformation_matrix[:-1, :],
                               dsize=(self.patch_size_final_resize, self.patch_size_final_resize),
                               flags=cv2.INTER_LANCZOS4)
        return image,transformation_matrix
