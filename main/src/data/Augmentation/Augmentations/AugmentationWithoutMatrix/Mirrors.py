"""Class computing random mirrors along a vertical or horizontal axis"""

from typing import Tuple

import numpy as np

from main.src.data.Augmentation.Augmentations.AugmentationWithoutMatrix.AbstractAugmentation import AbstractAugmentation


class Mirrors(AbstractAugmentation):
    """Class computing random mirrors along a vertical or horizontal axis
    Usage:

        >>> array = ...
        >>> annotation = ...
        >>> transformed_array, transformed_annotation = Mirrors().compute_random_augment(array,annotation)
        ... # Compute the random transformation with the static class
    """

    def compute_random_augment(self, image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the random mirrors transformations

        Args:
            image: np.ndarray, the original image to transform
            annotation: np.array, the corresponding annotation

        Returns:
            the transformed arrays (with the same transformations for image and annotation)

        """
        # Choose if compute lr mirror (mirror allong an vertical axis)
        if np.random.choice([True, False]) is True:
            image = np.fliplr(image)
            annotation = np.fliplr(annotation)
        # Choose if compute ud mirror (mirror allong an horizontal axis)
        if np.random.choice([True, False]) is True:
            image = np.flipud(image)
            annotation = np.flipud(annotation)
        return image, annotation
