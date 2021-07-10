from typing import Tuple, List

import numpy as np

from main.src.data.Augmentation.Augmentations.RotationResizeMirrors import RotationResizeMirrors
from main.src.param_savers.BaseClass import BaseClass


class Augmenter1(BaseClass):
    """Manage and keep track of augmentations to apply on source images only to directly extract patches

    With this class, only one augmentation is supported combinedRotResizeMir which allows to commpute the final patch to be provided to the attr_model after
    rotation, mirrors, resizes (one for augmentation and another to resize the patch to a smaller version)

        Args:
            allowed_transformations: str, augmentations to apply. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float}
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the attr_model

    """

    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int, allowed_transformations):
        self.attr_allowed_transformations = allowed_transformations
        self.attr_transformations_class = None
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        self.add_transformation(allowed_transformations, patch_size_before_final_resize, patch_size_final_resize)
        self.attr_augmented_dataset_parameters = {}

    def add_transformation(self, allowed_transformations: str, patch_size_before_final_resize: int,
                           patch_size_final_resize: int):
        """Method that map transformation names with actual classes.

        Args:
            allowed_transformations: str, list of augmentations to apply. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float}
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the attr_model
        Returns:

        """
        if "combinedRotResizeMir" in allowed_transformations:
            seen = True
            [_, rotation_step, resize_lower_fact_float, resize_upper_fact_float] = allowed_transformations.split("_")
            rotation_step = float(rotation_step)
            resize_lower_fact_float = float(resize_lower_fact_float)
            resize_upper_fact_float = float(resize_upper_fact_float)
            self.attr_transformations_classes = RotationResizeMirrors(rotation_step=rotation_step,
                                                                      resize_lower_fact_float=resize_lower_fact_float,
                                                                      resize_upper_fact_float=resize_upper_fact_float,
                                                                      patch_size_before_final_resize=patch_size_before_final_resize,
                                                                      patch_size_final_resize=patch_size_final_resize
                                                                      )
        else:
            raise NotImplementedError(f"{allowed_transformations} is not implemented")

    def transform(self, image: np.ndarray, annotation: np.ndarray, partial_transformation_matrix: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int, int], ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the random augmentations in the order in which they have been supplied.

                Apply the same augmentations to the image and to the annotation

                Args:
                    image: np.ndarray, the input image to transform
                    annotation: np.array, the corresponding annotation
                    partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
                    patch_upper_left_corner_coords: tuple of int, coordinates of the upperleft corner of the patch in the transformed image

                Returns:
                    tuple of 3 np.ndarray
                    - the transformed image patch
                    - the transformed annotation patch
                    - the transformation matrix

                """
        image, annotation, partial_transformation_matrix = self.attr_transformations_classes.compute_random_augment(
            image, annotation, partial_transformation_matrix,
            coord_patch=patch_upper_left_corner_coords)
        return image, annotation, partial_transformation_matrix

    def get_grid(self, img_shape, partial_transformation_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """

        return self.attr_transformations_classes.get_grid(img_shape, partial_transformation_matrix)

    def choose_new_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Method that allows to create a new augmentation dict containing

        Returns: np.ndarray, transformation matrix to apply the augmentation. It will be further required to "add" (dot multiply) the shift matrix to extract the correct patch
            ⚠️⚠️⚠️⚠️️ coordinates in OPENCV are in the opposite way of the normal row,cols way ⚠️⚠️⚠️⚠
            Internally this matrix include the following transformations:
            - angle
            - resize_factor
            - mirrorlr
            - mirrorud
        """
        return self.attr_transformations_classes.choose_new_augmentation(image)
