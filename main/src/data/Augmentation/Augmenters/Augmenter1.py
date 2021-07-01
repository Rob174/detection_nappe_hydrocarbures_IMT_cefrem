from main.src.data.Augmentation.Augmentations.Mirrors import Mirrors
from main.src.data.Augmentation.Augmentations.Resize import Resize
from main.src.data.Augmentation.Augmentations.RotationResizeMirrors import RotationResizeMirrors
from main.src.data.Augmentation.Augmentations.Rotations import Rotations
from main.src.data.Augmentation.Augmenters.Augmenter0 import Augmenter0
from main.src.param_savers.BaseClass import BaseClass

import numpy as np
from typing import Tuple, List, Any, Dict


class Augmenter1(BaseClass):
    """Manage and keep track of augmentations to apply on source images only to directly extract patches

    With this class, only one augmentation is supported combinedRotResizeMir which allows to commpute the final patch to be provided to the model after
    rotation, mirrors, resizes (one for augmentation and another to resize the patch to a smaller version)

        Args:
            allowed_transformations: str, augmentations to apply. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float}
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the model

    """
    def __init__(self, patch_size_before_final_resize: int, patch_size_final_resize: int, allowed_transformations):
        self.attr_allowed_transformations = allowed_transformations
        self.attr_transformations_class =  None
        self.attr_patch_size_before_final_resize = patch_size_before_final_resize
        self.attr_patch_size_final_resize = patch_size_final_resize
        self.add_transformation(allowed_transformations,patch_size_before_final_resize,patch_size_final_resize)
        self.attr_augmented_dataset_parameters = {}
    def add_transformation(self,allowed_transformations: str, patch_size_before_final_resize: int, patch_size_final_resize: int):
        """Method that map transformation names with actual classes.

        Args:
            allowed_transformations: str, list of augmentations to apply. Currently supported:
            - combinedRotResizeMir_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float}
            patch_size_before_final_resize: int, size in px of the output patch to extract
            patch_size_final_resize: int, size in px of the output patch provided to the model
        Returns:

        """
        if "combinedRotResizeMir" in allowed_transformations:
            seen = True
            [_,_,rotation_step,resize_lower_fact_float,resize_upper_fact_float] = allowed_transformations.split("_")
            rotation_step = float(rotation_step)
            resize_lower_fact_float = float(resize_lower_fact_float)
            resize_upper_fact_float = float(resize_upper_fact_float)
            self.attr_transformations_classes= RotationResizeMirrors(rotation_step=rotation_step,
                                                                           resize_lower_fact_float=resize_lower_fact_float,
                                                                           resize_upper_fact_float=resize_upper_fact_float,
                                                                           patch_size_before_final_resize=patch_size_before_final_resize,
                                                                           patch_size_final_resize=patch_size_final_resize
                                                                           )
        else:
            raise NotImplementedError(f"{allowed_transformations} is not implemented")
    def transform(self,image: np.ndarray, annotation: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int],**augmentation_parameters: Dict[str,Any]) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Compute the random augmentations in the order in which they have been supplied.

                Apply the same augmentations to the image and to the annotation

                Args:
                    image: np.ndarray, the input image to transform
                    annotation: np.array, the corresponding annotation
                    patch_upper_left_corner_coords: tuple of int, coordinates of the upperleft corner of the patch in the transformed image
                    augmentation_parameters: dict

                Returns:
                    tuple of 3 np.ndarray
                    - the transformed image patch
                    - the transformed annotation patch
                    - the transformation matrix

                """
        transformation_matrix = np.identity(3)
        image, annotation, partial_transformation_matrix = self.attr_transformations_classes.compute_random_augment(image, annotation,**augmentation_parameters)
        transformation_matrix = partial_transformation_matrix.dot(transformation_matrix)
        return image, annotation, transformation_matrix

    def choose_new_augmentations(self):
        return self.attr_transformations_classes.choose_new_augmentation()