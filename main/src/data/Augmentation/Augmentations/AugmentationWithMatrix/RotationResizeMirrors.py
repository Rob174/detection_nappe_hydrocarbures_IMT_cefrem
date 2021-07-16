from typing import Tuple, List, Callable

import cv2
import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationWithMatrix.AbstractAugmentationWithMatrix import \
    AbstractAugmentationWithMatrix


class RotationResizeMirrors(AbstractAugmentationWithMatrix):
    """Class computing random rotation along a vertical or horizontal axis

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

    def compute_image_augment(self, image: np.ndarray,
                              partial_transformation_matrix: np.ndarray,
                              coord_patch: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the random transformations at once on the image

        Args:
            image: np.ndarray, the original image to transform
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
            coord_patch: coordinates of the output patch in the augmented image

        Returns:
            tuple of 2 np.ndarray
            - the transformed image patch
            - the transformation matrix used
        """

        shift_patch_into_position_matrix = np.array([[1, 0, -coord_patch[1]],
                                                     [0, 1, -coord_patch[0]],
                                                     [0, 0, 1]])
        transformation_matrix = shift_patch_into_position_matrix.dot(partial_transformation_matrix)
        patch_image = cv2.warpAffine(image, transformation_matrix[:-1, :],
                                     dsize=(self.attr_patch_size_final_resize, self.attr_patch_size_final_resize),
                                     flags=cv2.INTER_LANCZOS4)
        return patch_image, transformation_matrix

    def compute_label_augment(self, annotation_function: Callable, image_name: str,
                              partial_transformation_matrix: np.ndarray,
                              coord_patch: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the random mirrors transformations at once on the annotation **points** directly

        It is indeed the only way to avoid new classes introductionn due to interpolation

        Args:
            annotation_function: Callable that can generate the patch with the given parameters
            image_name: str, the name of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
            coord_patch: coordinates of the output patch in the augmented image

        Returns:
            tuple of np.ndarray
            - the transformed annotation patch
            - the transformation matrix used
        """

        shift_patch_into_position_matrix = np.array([[1, 0, -coord_patch[1]],
                                                     [0, 1, -coord_patch[0]],
                                                     [0, 0, 1]])
        transformation_matrix = shift_patch_into_position_matrix.dot(partial_transformation_matrix)
        annotation = annotation_function(image_name, transformation_matrix, self.attr_patch_size_final_resize)
        return annotation, transformation_matrix

    def get_grid(self, img_shape, partial_transformation_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """
        rows, cols = img_shape[:2]
        original_mapped_corner1 = partial_transformation_matrix.dot([cols, rows, 1])
        original_mapped_corner2 = partial_transformation_matrix.dot([cols, rows, 1])
        max_rows = max(original_mapped_corner1[1], original_mapped_corner2[1]) - self.attr_patch_size_final_resize
        max_cols = max(original_mapped_corner1[0], original_mapped_corner2[0]) - self.attr_patch_size_final_resize
        cols_coords = np.arange(0, max_cols, self.attr_patch_size_final_resize)
        rows_coords = np.arange(0, max_rows, self.attr_patch_size_final_resize)
        coords = list(zip(*list(x.flat for x in np.meshgrid(rows_coords, cols_coords))))
        return coords

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

        src_points = np.array([[0, 0], [0, rows - 1], [cols - 1, 0], [cols - 1, rows - 1]], dtype=np.float32)
        dst_points = src_points
        if mirror == 0:
            dst_points = np.array([src_points[2], src_points[3], src_points[0], src_points[1]], dtype=np.float32)
        elif mirror == 1:
            dst_points = np.array([src_points[1], src_points[0], src_points[3], src_points[2]], dtype=np.float32)
        mirror_matrix = np.concatenate((cv2.getAffineTransform(src_points[:3], dst_points[:3]), [[0, 0, 1]]), axis=0)
        partial_transformation_matrix = (mirror_matrix.dot(partial_transformation_matrix))
        resize_matrix = np.array([[resize_factor, 0, 0],
                                  [0, resize_factor, 0],
                                  [0, 0, 1]])
        partial_transformation_matrix = resize_matrix.dot(partial_transformation_matrix)
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
