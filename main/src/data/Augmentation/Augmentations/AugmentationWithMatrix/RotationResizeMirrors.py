"""Class computing random rotation mirrors resize at once"""

from typing import Tuple, List, Callable

import cv2
import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationWithMatrix.AbstractAugmentationWithMatrix import \
    AbstractAugmentationWithMatrix


class RotationResizeMirrors(AbstractAugmentationWithMatrix):
    """Class computing random rotation mirrors resize

    Args:
        rotation_step: float, rotation step to take angle between angle 0 and 360° with a ... angle step
        resize_lower_fact_float: float, minimal resize factor to resize the original image
        resize_upper_fact_float: float, maximal resize factor to resize the original image
        patch_size_before_final_resize: int, size in px of the output patch to extract
        patch_size_final_resize: int, size in px of the output patch provided to the attr_model

    Usage:

        >>> image = ...
        >>> annotation = ...
        >>> augmentation = RotationResizeMirrors(patch_size_before_final_resize=1000,
        ...                                      patch_size_final_resize=256,rotation_step=15,
        ...                                      resize_lower_fact_float=0.25,
        ...                                      resize_upper_fact_float=4)
        >>> partial_transformation_matrix = augmentation.choose_new_augmentation(image.shape)
        >>> for coord in augmentation.get_grid(image.shape,partial_transformation_matrix):
        ...     patch_array, transformation_matrix = augmentation.compute_image_augment(image,
        ...                                                                              partial_transformation_matrix,
        ...                                                                              coord)
        ...     patch_annotation, _ = augmentation.compute_image_augment(image, partial_transformation_matrix,
        ...                                                              coord)
        ... # Compute the random transformation with the static class
        >>> patch_array.shape
        (256,256)
    """

    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int, rotation_step: float,
                 resize_lower_fact_float: float, resize_upper_fact_float: float):
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        self.attr_rotation_step = rotation_step
        self.attr_resize_upper_fact_float = resize_upper_fact_float
        self.attr_resize_lower_fact_float = resize_lower_fact_float


    def choose_parameters(self) -> Tuple[float, float, int]:
        """Choose random parameters for augmentations

        Returns: tuple (angle,resize_factor,mirror)
            angle, float angle of rotation
            resize_factor, float resize factor taking into account the final resize to get the input image for the model
            mirror, int 0 = fliplr ; 1 = flipud ; -1 = noflip

        """
        angle = np.random.choice(np.arange(0, 361, self.attr_rotation_step))
        resize_factor = np.random.rand() * (
                self.attr_resize_upper_fact_float - self.attr_resize_lower_fact_float) + self.attr_resize_lower_fact_float
        resize_factor *= self.attr_patch_size_final_resize / self.attr_patch_size_before_final_resize

        mirror = np.random.choice([0, 1, -1])
        return angle, resize_factor, mirror

    def compute_transformation_matrix(self, rows: int, cols: int, angle: float, resize_factor: float,
                                      mirror: int) -> np.ndarray:
        """ Compute the transformation matrix corresponding to the parameters supplied

        Args:
            rows: number of rows of the input image
            cols: number of cols of the input image
            angle, float angle of rotation
            resize_factor, float resize factor taking into account the final resize to get the input image for the model
            mirror, int 0 = fliplr ; 1 = flipud ; -1 = noflip

        Returns:
            np.ndarray, the transformation matrix
        """
        # Transformation matrix construction  ⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️ coordinates in OPENCV are in the opposite way of the normal row,cols way
        partial_transformation_matrix = np.identity(3)
        # Mirrors
        src_points = np.array([[0, 0], [0, rows - 1], [cols - 1, 0], [cols - 1, rows - 1]], dtype=np.float32)
        dst_points = src_points
        if mirror == 0:
            dst_points = np.array([src_points[2], src_points[3], src_points[0], src_points[1]], dtype=np.float32)
        elif mirror == 1:
            dst_points = np.array([src_points[1], src_points[0], src_points[3], src_points[2]], dtype=np.float32)
        mirror_matrix = np.concatenate((cv2.getAffineTransform(src_points[:3], dst_points[:3]), [[0, 0, 1]]), axis=0)
        partial_transformation_matrix = (mirror_matrix.dot(partial_transformation_matrix))
        # Resize
        resize_matrix = np.array([[resize_factor, 0, 0],
                                  [0, resize_factor, 0],
                                  [0, 0, 1]])
        partial_transformation_matrix = resize_matrix.dot(partial_transformation_matrix)
        # Rotate
        rotate_matrix = np.concatenate((cv2.getRotationMatrix2D((cols * resize_factor / 2, rows * resize_factor / 2),
                                                                angle=angle, scale=1), [[0, 0, 1]]), axis=0)
        partial_transformation_matrix = rotate_matrix.dot(partial_transformation_matrix)
        adjusted_translation = np.array(
            [[1, 0., -min(0, partial_transformation_matrix.dot([cols - 1, rows - 1, 1])[0])],
             [0, 1, -min(0, partial_transformation_matrix.dot([cols - 1, rows - 1, 1])[1])],
             [0, 0, 1]])
        partial_transformation_matrix = adjusted_translation.dot(partial_transformation_matrix)
        return partial_transformation_matrix

    def choose_new_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Method that allows to create a new augmentation dict containing

        Returns: np.ndarray, transformation matrix to apply the augmentation. It will be further required to "add" (dot multiply) the shift matrix to extract the correct patch
            ⚠️⚠️⚠️⚠️️ coordinates in OPENCV are in the opposite way of the normal row,cols way ⚠️⚠️⚠️⚠
            Internally this matrix include the following transformations:
            - angle
            - resize_factor
            - mirrorlr
            - mirrorud
        """
        # Choose parameters of transformation if not already chosen for epoch item
        rows, cols = image.shape[:2]

        return self.compute_transformation_matrix(rows, cols, *self.choose_parameters())
