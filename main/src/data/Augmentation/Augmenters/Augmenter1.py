from main.src.data.Augmentation.Augmentations.Mirrors import Mirrors
from main.src.data.Augmentation.Augmentations.Resize import Resize
from main.src.data.Augmentation.Augmentations.RotationResizeMirrors import RotationResizeMirrors
from main.src.data.Augmentation.Augmentations.Rotations import Rotations
from main.src.data.Augmentation.Augmenters.Augmenter0 import Augmenter0
from main.src.param_savers.BaseClass import BaseClass

import numpy as np
from typing import Tuple, List


class Augmenter1(BaseClass):
    """Manage and keep track of augmentations to apply on source images only to directly extract patches

        Args:
            allowed_transformations: str, list of augmentations to apply seprated by commas. Currently supported:
            - mirrors
            - rotations
            - resize: provided as "resize_...range..._...shift..."
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the model

    """
    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int, allowed_transformations="mirrors,rotations"):
        self.attr_allowed_transformations = allowed_transformations
        self.attr_transformations_classes =  []
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        self.add_transformation(allowed_transformations,patch_size_before_final_resize,patch_size_final_resize)
    def add_transformation(self,allowed_transformations: str, patch_size_before_final_resize: int, patch_size_final_resize: int):
        """Method that map transformation names with actual classes. Splited from the __init__ to be able to overload

        Args:
            allowed_transformations: str, list of augmentations to apply seprated by commas. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float} (has to be selected to extract the patch
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the model
        Returns:

        """
        for transfo in allowed_transformations.split(","):
            if "combinedRotResizeMir" in transfo:
                seen = True
                [_,_,rotation_step,resize_lower_fact_float,resize_upper_fact_float] = transfo.split("_")
                rotation_step = float(rotation_step)
                resize_lower_fact_float = float(resize_lower_fact_float)
                resize_upper_fact_float = float(resize_upper_fact_float)
                self.attr_transformations_classes.append(RotationResizeMirrors(rotation_step=rotation_step,
                                                                               resize_lower_fact_float=resize_lower_fact_float,
                                                                               resize_upper_fact_float=resize_upper_fact_float,
                                                                               patch_size_before_final_resize=patch_size_before_final_resize,
                                                                               patch_size_final_resize=patch_size_final_resize
                                                                               ))
            else:
                raise NotImplementedError(f"{transfo} is not implemented")
    def transform(self,image: np.ndarray, annotation: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Compute the random augmentations in the order in which they have been supplied.

                Apply the same augmentations to the image and to the annotation

                Args:
                    image: np.ndarray, the input image to transform
                    annotation: np.array, the corresponding annotation
                    patch_upper_left_corner_coords: tuple of int, coordinates of the upperleft corner of the patch in the transformed image

                Returns:
                    The randomly transformed image and annotation with the global transformation matrix

                """
        transformation_matrix = np.identity(3)
        for transfoObj in self.attr_transformations_classes:
            image, annotation, partial_transformation_matrix = transfoObj.compute_random_augment(image, annotation)
            transformation_matrix = partial_transformation_matrix.dot(transformation_matrix)
        return image, annotation, transformation_matrix