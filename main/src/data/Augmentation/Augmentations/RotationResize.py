from main.src.param_savers.BaseClass import BaseClass
import numpy as np
from typing import Tuple
import cv2

class RotationResize(BaseClass):
    """Class computing random rotation along a vertical or horizontal axis

    Args:
        rotation_step: float, rotation step to take angle between angle 0 and 360° with a ... angle step
        resize_lower_fact_float: float, minimal resize factor to resize the original image
        resize_upper_fact_float: float, maximal resize factor to resize the original image
        patch_size: int, size of the patch in px

    Usage:

        >>> array = ...
        >>> annotation = ...
        >>> patch_array, patch_annotation = RotationResize(rotation_step=15,resize_lower_fact_float=0.25,resize_upper_fact_float=4,patch_size=1000).compute_random_augment(array,annotation)
        ... # Compute the random transformation with the static class
        >>> patch_array
    """
    def __init__(self,rotation_step: float, resize_lower_fact_float: float, resize_upper_fact_float: float, patch_size: int):
        self.attr_angle_step = rotation_step
        self.attr_resize_lower_fact_float = resize_lower_fact_float
        self.attr_resize_upper_fact_float = resize_upper_fact_float
        self.attr_patch_size = patch_size
    def compute_random_augment(self,image: np.ndarray, annotation: np.ndarray,
                               coord_patch: Tuple[int], size_patch: int) -> Tuple[np.ndarray,np.ndarray]:
        """Compute the random mirrors transformations

        Args:
            image: np.ndarray, the original image to transform (
            annotation: np.array, the corresponding annotation

        Returns:
            the transformed arrays (with the same transformations for image and annotation)

        """
        # Choose of which angle (in degrees) we want to rotate the input image
        angle = np.random.choice(np.arange(0,361,self.attr_angle_step))
        resize_factor = np.random.rand()*(self.attr_resize_upper_fact_float-self.attr_resize_upper_fact_float)+self.attr_resize_lower_fact_float

        resize_matrix = np.array([[resize_factor, 0,             0],
                                  [0,             resize_factor, 0],
                                  [0,             0,             1]])
        rows,cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((rows/2,cols/2),angle,scale=1) # because matrix (2,3) we make...
        rotation_matrix = np.concatenate((rotation_matrix,[[0,0,1]]),axis=0) # matrix (3,3)
        shift_patch_into_position_matrix = np.array([[1,    0,  -coord_patch[0]],
                                                     [0,    1,  -coord_patch[1]],
                                                     [0,    0,   1]])
        transformation_matrix = shift_patch_into_position_matrix.dot(resize_matrix.dot(rotation_matrix))
        patch_image = cv2.warpAffine(image,transformation_matrix,dsize=(size_patch,size_patch))
        patch_annotation = cv2.warpAffine(annotation,transformation_matrix,dsize=(size_patch,size_patch))
        return patch_image,patch_annotation