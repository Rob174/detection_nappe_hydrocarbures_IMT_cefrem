from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Callable

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class AbstractAugmentationWithMatrix(ABC, BaseClass):
    @abstractmethod
    def compute_image_augment(self, image: np.ndarray,
                              partial_transformation_matrix: np.ndarray,
                              coord_patch: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Make the transformation on the annotation and returns the results with the affine transformation matrix"""
        pass

    @abstractmethod
    def compute_label_augment(self, annotation_function: Callable, image_name: str,
                              partial_transformation_matrix: np.ndarray,
                              coord_patch: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Make the transformation on the annotation and returns the results with the affine transformation matrix"""
        pass

    def choose_new_augmentation(self, image: np.ndarray) -> Dict[str, Any]:
        """Method that allows to create a new augmentation dict containing augmentation informations"""
        pass
