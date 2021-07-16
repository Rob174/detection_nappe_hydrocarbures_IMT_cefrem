from typing import Tuple

import cv2
import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationWithoutMatrix.AbstractAugmentation import AbstractAugmentation


class Resize(AbstractAugmentation):
    def __init__(self, range: float, shift: float):
        self.attr_range = range
        self.attr_shift = shift

    def compute_random_augment(self, image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the resize transformation on the inputs"""
        factor = np.random.rand() * (self.attr_range - self.attr_shift) + self.attr_shift
        image = cv2.resize(image, dsize=image.shape, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        annotation = cv2.resize(annotation, dsize=annotation.shape, fx=factor, fy=factor,
                                interpolation=cv2.INTER_LANCZOS4)
        return image, annotation
