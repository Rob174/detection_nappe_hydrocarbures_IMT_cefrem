"""Apply no augmentations in the case of step by step augmentations (same procedure as Augmenter0 not Augmenter1)"""

from typing import Tuple

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class NoAugmenter(BaseClass):
    """Manage and keep track of augmentations to apply (here none)
    """

    def __init__(self, *args, **kargs):
        self.attr_allowed_transformations = "none"

    def transform(self, image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute no transformation but allows to keep the constant class format

        Args:
            image: np.ndarray, the input image to transform
            annotation: np.array, the corresponding annotation

        Returns:
            The randomly transformed image and annotation

        """
        return image, annotation
