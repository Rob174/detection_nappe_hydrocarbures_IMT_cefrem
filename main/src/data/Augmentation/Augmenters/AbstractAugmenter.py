"""BaseClass to build an augmenter which manages augmentations"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any

import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationApplier.AbstractApplier import AbstractApplier
from main.src.data.GridMaker.AbstractGridMaker import AbstractGridMaker


class AbstractAugmenter(ABC):
    """BaseClass to build an augmenter which manages augmentations

    Args:
        allowed_transformations: List[str], augmentations to apply.
        patch_size_before_final_resize: int, size in px of the output patch to extract
        patch_size_final_resize: int, size in px of the output patch provided to the attr_model
    """

    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int,
                 allowed_transformations: List[str]):
        self.attr_allowed_transformations = allowed_transformations
        self.attr_transformations_class = None
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        for transformation in allowed_transformations:
            self.add_transformation(transformation, patch_size_before_final_resize, patch_size_final_resize)

    @property
    def image_applier(self) -> AbstractApplier:
        raise NotImplementedError

    @property
    def label_applier(self) -> AbstractApplier:
        raise NotImplementedError

    @property
    def grid_maker(self) -> AbstractGridMaker:
        raise NotImplementedError

    @abstractmethod
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
        pass

    @abstractmethod
    def transform_image(self, image: np.ndarray, partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the random augmentations in the order in which they have been supplied.

                Apply the same augmentations to the image and to the annotation

                Args:
                    image: np.ndarray, the input image to transform
                    annotation: np.array, the corresponding annotation
                    partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
                    patch_upper_left_corner_coords: tuple of int, coordinates of the upperleft corner of the patch in the transformed image

                Returns:
                    tuple of 2 np.ndarray
                    - the transformed image patch*
                    - the transformation matrix

                """
        pass

    def transform_label(self, data: Any, partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute no augmentations on points

        It is indeed the only way to avoid new classes introductionn due to interpolation

        Args:
            annotation_function: Callable that can generate the patch with the given parameters
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
            coord_patch: coordinates of the output patch in the augmented image

        Returns:
            tuple of np.ndarray
            - the annotation patch
            - the transformation matrix used (with patch extraction)
        """
        return self.label_applier.transform(data, partial_transformation_matrix,
                                            patch_upper_left_corner_coords)

    def get_grid(self, img_shape, partial_transformation_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """

        return self.grid_maker.get_grid(img_shape, partial_transformation_matrix)

    def choose_new_augmentations(self, image: np.ndarray) -> np.ndarray:
        """Method that allows to create a new augmentation dict containing

        Returns: np.ndarray, transformation matrix to apply the augmentation. It will be further required to "add" (dot multiply) the shift matrix to extract the correct patch
            ⚠️⚠️⚠️⚠️️ coordinates in OPENCV are in the opposite way of the normal row,cols way ⚠️⚠️⚠️⚠
            Internally this matrix include the following transformations:
            - resize_factor: to get the correct final patch size
        """
        resize_factor = self.attr_patch_size_final_resize / self.attr_patch_size_before_final_resize
        return np.array([[resize_factor, 0, 0], [0, resize_factor, 0], [0, 0, 1]], dtype=np.float32)
