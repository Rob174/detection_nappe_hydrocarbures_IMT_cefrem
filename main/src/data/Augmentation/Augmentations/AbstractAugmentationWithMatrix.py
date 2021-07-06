from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

from main.src.param_savers.BaseClass import BaseClass


class AbstractAugmentationWithMatrix(ABC, BaseClass):
    @abstractmethod
    def compute_random_augment(self, image: np.ndarray, annotation: np.ndarray,
                               partial_transformation_matrix: np.ndarray,
                               coord_patch: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make the transformation on the image and annotation and returns the results with the affine transformation matrix"""
        pass

    def choose_new_augmentation(self, image: np.ndarray) -> Dict[str,Any]:
        """Method that allows to create a new augmentation dict containing augmentation informations"""
        pass