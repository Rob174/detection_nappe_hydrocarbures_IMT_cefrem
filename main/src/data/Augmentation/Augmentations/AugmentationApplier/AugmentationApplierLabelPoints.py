import cv2
import numpy as np
from typing import Tuple, Callable

from main.src.data.GridMaker.GridMaker import GridMaker
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
    def transform(self,image_name: str, partial_transformation_matrix: np.ndarray,
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
        label = self.access_function(image_name, transformation_matrix, self.patch_size_final_resize)
        return label,transformation_matrix
