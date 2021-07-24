"""Apply all transformations at once thanks to the transformation matrix and warpAffine. """

from typing import Tuple, List, Callable, Dict

import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationApplier.AugmentationApplierImage import \
    AugmentationApplierImage
from main.src.data.Augmentation.Augmentations.AugmentationApplier.AugmentationApplierLabelPoints import \
    AugmentationApplierLabelPoints
from main.src.data.Augmentation.Augmentations.AugmentationWithMatrix.RotationResizeMirrors import RotationResizeMirrors
from main.src.data.Augmentation.Augmenters.AbstractAugmenter import AbstractAugmenter
from main.src.data.GridMaker.GridMaker import GridMaker
from main.src.param_savers.BaseClass import BaseClass


class Augmenter1(BaseClass, AbstractAugmenter):
    """Manage and keep track of augmentations to apply on source images only to directly extract patches

    With this class, only one augmentation is supported combinedRotResizeMir which allows to commpute the final patch to be provided to the attr_model after
    rotation, mirrors, resizes (one for augmentation and another to resize the patch to a smaller version)

    This class splits annotation generation and image generation.
    It allows to filter the global sample on the label as it costs less to generate it

        Args:
            allowed_transformations: List[str], augmentations to apply. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float}
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the attr_model
            label_access_function:Callable[[str,np.ndarray,int],np.ndarray], function to accesss the points
    """

    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int,
                 allowed_transformations: List[str]):
        super(Augmenter1, self).__init__(patch_size_before_final_resize, patch_size_final_resize,
                                         allowed_transformations)
        self.attr_grid_maker = GridMaker(patch_size_final_resize=patch_size_final_resize)
        self.attr_label_applier = AugmentationApplierLabelPoints(
            grid_maker=self.attr_grid_maker,
            patch_size_final_resize=patch_size_final_resize
        )
        self.attr_image_applier = AugmentationApplierImage(
            grid_maker=self.attr_grid_maker,
            patch_size_final_resize=patch_size_final_resize
        )

    @property
    def grid_maker(self) -> GridMaker:
        return self.attr_grid_maker

    @property
    def label_applier(self) -> AugmentationApplierLabelPoints:
        return self.attr_label_applier

    @property
    def image_applier(self) -> AugmentationApplierImage:
        return self.attr_image_applier

    def set_image_access_function(self, function: Callable[[str], Tuple[np.ndarray, np.ndarray]]):
        self.attr_image_applier.access_function = function

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
        return self.attr_image_applier.transform(image, partial_transformation_matrix, patch_upper_left_corner_coords)

    def transform_label(self, data: List[Dict], partial_transformation_matrix: np.ndarray,
                        patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the random mirrors transformations at once on the annotation **points** directly

        It is indeed the only way to avoid new classes introductionn due to interpolation

        Args:
            data: List[Dict] List of the polygons on the patch
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)
            coord_patch: coordinates of the output patch in the augmented image

        Returns:
            tuple of np.ndarray
            - the transformed annotation patch
            - the transformation matrix used
        """
        return self.attr_label_applier.transform(data, partial_transformation_matrix, patch_upper_left_corner_coords)

    def get_grid(self, img_shape, partial_transformation_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """

        return self.grid_maker.get_grid(img_shape, partial_transformation_matrix)

    def compute_transformation_matrix(self, rows, cols, angle, resize_factor, mirror):
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
        return self.attr_transformations_classes.compute_transformation_matrix(rows, cols, angle, resize_factor, mirror)

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
