"""Base class to construct a new standardizer"""

import numpy as np


class AbstractStandardizer:
    def __init__(self, *args, **kwargs):
        pass

    @property
    def mean(self):
        """Mean of the dataset"""
        raise NotImplementedError

    @property
    def std(self):
        """Standard deviation of the dataset"""
        raise NotImplementedError

    @property
    def n(self) -> int:
        """Number of samples in dataset"""
        raise NotImplementedError

    def standardize(self, image: np.ndarray):
        """Standardize the input array"""
        raise NotImplementedError
