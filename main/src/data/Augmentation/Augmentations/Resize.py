import numpy as np
import cv2
from typing import Tuple


class Resize:

    @staticmethod
    def compute_random_augment(image: np.ndarray, annotation: np.ndarray, range: float, shift: float) -> Tuple[np.ndarray,np.ndarray]:
        factor = np.random.rand()*(range-shift)+shift
        image = cv2.resize(image,dsize=image.shape,fx=factor,fy=factor,interpolation=cv2.INTER_LANCZOS4)
        annotation = cv2.resize(annotation, dsize=annotation.shape, fx=factor, fy=factor, interpolation=cv2.INTER_LANCZOS4)
        return image,annotation