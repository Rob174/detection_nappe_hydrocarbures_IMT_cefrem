"""Abstract class to create new class adder"""

from abc import ABC
from typing import Optional, Tuple

import numpy as np


class AbstractClassAdder(ABC):
    """Base class to build a new class adder"""

    def __init__(self, interval: int):
        self.attr_interval = interval

    def generate_if_required(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """Method that generates a sample if it is the turn of the patch adder based on the interval to wait provided in the constructor

        Returns:
            Optional[Tuple]
            - if it is this dataset turn: patch_image, patch_annotation, transformation_matrix (used to build the patches), source image name
            - else None
        """
        return None

    def set_interval(self, interval: int):
        """Method setter to dynamically change the interval if we want to complexify the training by reducing more and more the interval"""
        self.attr_interval = interval
