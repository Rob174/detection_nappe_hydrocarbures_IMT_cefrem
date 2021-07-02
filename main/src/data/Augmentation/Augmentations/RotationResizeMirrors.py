from main.src.param_savers.BaseClass import BaseClass
import numpy as np
from typing import Tuple, Sequence, Iterator
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

        >>> image = ...
        >>> annotation = ...
        >>> augmentation = RotationResizeMirrors(patch_size_before_final_resize=1000,
        ...                                      patch_size_final_resize=256,rotation_step=15,
        ...                                      resize_lower_fact_float=0.25,
        ...                                      resize_upper_fact_float=4)
        >>> partial_transformation_matrix = augmentation.choose_new_augmentation(image.shape)
        >>> for coord in augmentation.get_grid(image.shape,partial_transformation_matrix):
        ...     patch_array, patch_annotation, transformation_matrix = augmentation.compute_random_augment(image,annotation,
        ...                                                                                                partial_transformation_matrix,
        ...                                                                                                coord)
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
    def compute_random_augment(self,image: np.ndarray, annotation: np.ndarray, partial_transformation_matrix: np.ndarray,
                               coord_patch: Tuple[int,int]) -> Tuple[np.ndarray,np.ndarray, np.ndarray]:
        """Compute the random mirrors transformations

        Args:
            image: np.ndarray, the original image to transform (
            annotation: np.array, the corresponding annotation
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
            coord_patch: coordinates of the output patch in the augmented image

        Returns:
            tuple of 3 np.ndarray
            - the transformed image patch
            - the transformed annotation patch
            - the transformation matrix
        """

        shift_patch_into_position_matrix = np.array([[1,    0,  -coord_patch[0]],
                                                     [0,    1,  -coord_patch[1]],
                                                     [0,    0,   1]])
        transformation_matrix = shift_patch_into_position_matrix.dot(partial_transformation_matrix)
        patch_image = cv2.warpAffine(image,transformation_matrix[:-1,:],dsize=(self.attr_patch_size_final_resize,self.attr_patch_size_final_resize))
        patch_annotation = cv2.warpAffine(annotation,transformation_matrix[:-1,:],dsize=(self.attr_patch_size_final_resize,self.attr_patch_size_final_resize))
        return patch_image,patch_annotation, transformation_matrix
    def get_grid(self,img_shape,partial_transformation_matrix: np.ndarray) -> Iterator[Tuple[int,int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """
        rows,cols = img_shape[:2]
        original_corners = np.array([[0,0,1],
                                    [rows-1,0,1],
                                    [0,cols-1,1],
                                    [rows-1,cols-1,1]])
        original_mapped_corners = list(map(lambda x:x.dot(partial_transformation_matrix),original_corners))
        x_coords = list(map(lambda x:x[0],original_mapped_corners))
        y_coords = list(map(lambda x: x[1],original_mapped_corners))
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        x_coords = np.arange(min_x,max_x,self.attr_patch_size_final_resize)
        y_coords = np.arange(min_y, max_y,self.attr_patch_size_final_resize)
        iterator_coords = zip(*list(x.flat for x in np.meshgrid(x_coords, y_coords)))
        return iterator_coords

    def choose_new_augmentation(self,img_shape):
        """Method that allows to create a new augmentation dict containing

        Returns: np.ndarray, transformation matrix to apply the augmentation. It will be further required to "add" (dot multiply) the shift matrix to extract the correct patch
            Internally this matrix include the following transformations:
            - angle
            - resize_factor
            - mirrorlr
            - mirrorud
        """
        # Choose parameters of transformation if not already chosen for epoch item
        ## Individual parameters
        angle = np.random.choice(np.arange(0, 361, self.attr_rotation_step))
        resize_factor = np.random.rand() * (
                    self.attr_resize_upper_fact_float - self.attr_resize_lower_fact_float) + self.attr_resize_lower_fact_float
        resize_factor *= self.attr_patch_size_final_resize / self.attr_patch_size_before_final_resize

        mirrorlr = np.random.choice([1, -1])
        mirrorud = np.random.choice([1, -1])
        resize_matrix = np.array([[resize_factor * mirrorud, 0, 0],
                                  [0, resize_factor * mirrorlr, 0],
                                  [0, 0, 1]])
        rows, cols = img_shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((rows / 2, cols / 2), angle,
                                                  scale=1)  # because matrix (2,3) we make...
        rotation_matrix = np.concatenate((rotation_matrix, [[0, 0, 1]]), axis=0)  # matrix (3,3)
        partial_transformation_matrix = rotation_matrix.dot(resize_matrix)

        return partial_transformation_matrix