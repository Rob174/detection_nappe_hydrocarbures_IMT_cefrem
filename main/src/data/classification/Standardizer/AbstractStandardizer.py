
import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class AbstractStandardizer(BaseClass):
    def __init__(self,*args,**kwargs):
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
    def standardize(self,image: np.ndarray):
        """Standardize the input array"""
        raise NotImplementedError