"""To apply a transformation matrix on an image and directly taking the patch as np array"""

from typing import Tuple

import cv2
import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationApplier.AbstractApplier import AbstractApplier
from main.src.data.GridMaker.GridMaker import GridMaker
from main.src.param_savers.BaseClass import BaseClass


class AugmentationApplierImage(BaseClass, AbstractApplier):
    """To apply a transformation matrix on an image and directly taking the patch

    Args:
        grid_maker: GridMaker, to get the coordinates of the patches
        patch_size_final_resize: int, final size of the patch before giving it to the model
    """

    def __init__(self, grid_maker: GridMaker, patch_size_final_resize: int):
        super(AugmentationApplierImage, self).__init__(grid_maker, patch_size_final_resize)

    def transform(self, data: np.ndarray, partial_transformation_matrix: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transformation on the image

        Args:
            data: np.ndarray, image to transform
            partial_transformation_matrix: np.ndarray, transformation to apply without the translation to get the correct patch
            patch_upper_left_corner_coords: Tuple[int,int], coordinates of the upper left corner of the patch to take

        Returns:
            (image,transformation_matrix), tuple:
                - image: np.ndarray: resulting patch
                - transformation_matrix: np.ndarray, final transformation matrix applied
        """
        transformation_matrix = self.grid_maker.get_patch_transformation_matrix(
            patch_upper_left_corner_coords).dot(partial_transformation_matrix)
        image = cv2.warpAffine(data, transformation_matrix[:-1, :],
                               dsize=(self.patch_size_final_resize, self.patch_size_final_resize),
                               flags=cv2.INTER_LANCZOS4)
        return image, transformation_matrix
