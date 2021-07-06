import numpy as np
import cv2
from typing import Tuple

from main.src.param_savers.BaseClass import BaseClass


class Resize(BaseClass):
    def __init__(self, range: float, shift: float):
        self.attr_range = range
        self.attr_shift = shift

    def compute_random_augment(self, image: np.ndarray,
                               annotation: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        factor = np.random.rand() * (self.attr_range - self.attr_shift) + self.attr_shift
        image = cv2.resize(image, dsize=image.shape, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        annotation = cv2.resize(annotation, dsize=annotation.shape, fx=factor, fy=factor,
                                interpolation=cv2.INTER_LANCZOS4)
        return image, annotation
