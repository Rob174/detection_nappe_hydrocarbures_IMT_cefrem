"""Class to check if a patch is in the margin of the image based on the number of pixel to exactly 0"""
import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class MarginCheck(BaseClass):
    """Class to check if a patch is in the margin of the image based on the number of pixel to exactly 0
    """

    def __init__(self, threshold: int = 1000):
        """

        Args:
            threshold: int, number of pixel beyond which we consider that the patch is in the margin
        """
        self.attr_threshold = threshold

    def check_reject(self, patch: np.ndarray):
        """Check if a patch is in the margin

        Args:
            patch: np.ndarray of the raster image

        Returns:
            boolean, True if rejected False otherwise
        """
        if len(patch[patch == 0]) > self.attr_threshold:
            return True
        return False
