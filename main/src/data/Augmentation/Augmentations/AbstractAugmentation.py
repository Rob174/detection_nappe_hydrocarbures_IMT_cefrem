from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class AbstractAugmentation(ABC, BaseClass):
    @abstractmethod
    def compute_random_augment(self, image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the transformation on the inputs"""
        pass
