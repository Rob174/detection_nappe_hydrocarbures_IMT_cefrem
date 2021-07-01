from main.src.param_savers.BaseClass import BaseClass
import numpy as np
from typing import Tuple
import cv2

class RotationResizeMirrors(BaseClass):
    """Class computing random rotation along a vertical or horizontal axis

    Args:
        rotation_step: float, rotation step to take angle between angle 0 and 360° with a ... angle step
        resize_lower_fact_float: float, minimal resize factor to resize the original image
        resize_upper_fact_float: float, maximal resize factor to resize the original image
        patch_size_before_final_resize: int, size in px of the output patch to extract
        patch_size_final_resize: int, size in px of the output patch provided to the model

    Usage:

        >>> array = ...
        >>> annotation = ...
        >>> patch_array, patch_annotation, transformation_matrix = RotationResizeMirrors(patch_size_before_final_resize=1000, patch_size_final_resize=256,rotation_step=15,resize_lower_fact_float=0.25,resize_upper_fact_float=4).compute_random_augment(array,annotation,
        ...                                                                                        angle=15,resize_factor=2,
        ...                                                                                        mirrorlr=False,mirrorud=True,
        ...                                                                                        coord_patch=(1000,1000))
        ... # Compute the random transformation with the static class
        >>> patch_array.shape
        (256,256)
    """
    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int,rotation_step: float,resize_lower_fact_float: float,resize_upper_fact_float: float):
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        self.attr_rotation_step = rotation_step
        self.attr_resize_upper_fact_float = resize_upper_fact_float
        self.attr_resize_lower_fact_float = resize_lower_fact_float
    def compute_random_augment(self,image: np.ndarray, annotation: np.ndarray, angle: float, resize_factor: int, mirrorlr: bool, mirrorud: bool,
                               coord_patch: Tuple[int,int], *args, **kargs) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
        """Compute the random mirrors transformations

        Args:
            image: np.ndarray, the original image to transform (
            annotation: np.array, the corresponding annotation
            angle: float, angle of rotation to apply to the original image
            resize_factor: float resize factor to use ⚠️⚠️ do not choose a too small resize_factor not to make annotations disappear
            mirrorlr: bool, wether if we flip coordinates along the first axis
            mirrorud: bool, wether if we flip coordinates along the second axis
            coord_patch: coordinates of the output patch in the augmented image
            args: to exclude other eventual futur arguments for other classes
            kargs: to exclude other eventual futur arguments for other classes

        Returns:
            tuple of 3 np.ndarray
            - the transformed image patch
            - the transformed annotation patch
            - the transformation matrix
        """
        # Choose parameters of transformation if not already chosen for epoch item
        factor_x,factor_y = 1,1
        if mirrorlr is True:
            factor_y = -1
        if mirrorud is True:
            factor_x = -1
        resize_matrix = np.array([[resize_factor*factor_x,  0,                      0],
                                  [0,                       resize_factor*factor_y, 0],
                                  [0,                       0,                      1]])
        rows,cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((rows/2,cols/2),angle,scale=1) # because matrix (2,3) we make...
        rotation_matrix = np.concatenate((rotation_matrix,[[0,0,1]]),axis=0) # matrix (3,3)
        shift_patch_into_position_matrix = np.array([[1,    0,  -coord_patch[0]],
                                                     [0,    1,  -coord_patch[1]],
                                                     [0,    0,   1]])
        transformation_matrix = shift_patch_into_position_matrix.dot(resize_matrix.dot(rotation_matrix))
        patch_image = cv2.warpAffine(image,transformation_matrix,dsize=(self.attr_patch_size_final_resize,self.attr_patch_size_final_resize))
        patch_annotation = cv2.warpAffine(annotation,transformation_matrix,dsize=(self.attr_patch_size_final_resize,self.attr_patch_size_final_resize))
        return patch_image,patch_annotation, transformation_matrix

    def choose_new_augmentation(self):
        """Method that allows to create a new augmentation dict containing

        Returns: dict
            "angle": float
            "resize_factor": float
            "mirrorlr": bool
            "mirrorud": bool
        """
        dico_augmentations = {}
        dico_augmentations["angle"] = np.random.choice(np.arange(0,361,self.attr_rotation_step))
        dico_augmentations["resize_factor"] = np.random.rand()*(self.attr_resize_upper_fact_float-self.attr_resize_lower_fact_float) + self.attr_resize_lower_fact_float
        dico_augmentations["resize_factor"] *= self.attr_patch_size_final_resize / self.attr_patch_size_before_final_resize
        dico_augmentations["mirrorlr"] = np.random.choice([True,False])
        dico_augmentations["mirrorud"] = np.random.choice([True, False])
        return dico_augmentations