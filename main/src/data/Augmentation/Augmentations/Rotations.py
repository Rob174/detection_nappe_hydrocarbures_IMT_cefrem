import numpy as np
import cv2
from typing import Tuple

class Rotations:
    """Class computing random rotations of 15° steps
    Usage:

        >>> array = ...
        >>> annotation = ...
        >>> transformed_array, transformed_annotation = Rotations.compute_random_augment(array,annotation)
        ... # Compute the random transformation with the static class
    """
    attr_angle_step = 15
    @staticmethod
    def compute_random_augment(image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """Compute the random rotations transformations

        Args:
            image: np.ndarray, the original image to transform

        Returns:
            the transformed array

        """
        # Choose of which angle (in degrees) we want to rotate the input image
        angle = np.random.choice(np.arange(0,361,Rotations.attr_angle_step))
        if angle == 0:
            return image,annotation
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4)
        annotation = cv2.warpAffine(annotation, rotation_matrix, annotation.shape[1::-1],
                                flags=cv2.INTER_LANCZOS4)
        return image,annotation