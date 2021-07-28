"""BaseClass to build a Augmentation that returns the matrix of transformation"""

from abc import ABC

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class AbstractAugmentationWithMatrix(ABC, BaseClass):
    """BaseClass to build a Augmentation and returns the matrix of transformation"""

    def choose_new_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Method that allows to create a new augmentation returning the transformation matrix"""
        pass
