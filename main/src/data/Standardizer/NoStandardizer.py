"""No standardization applied"""

import numpy as np

from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.param_savers.BaseClass import BaseClass


class NoStandardizer(BaseClass, AbstractStandardizer):
    """Standardizer to apply when we use the OtherClassPatchAdder class

    Args:
        interval: interval between two patches with only the other class
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    @property
    def mean(self):
        raise NotImplementedError

    @property
    def std(self):
        raise NotImplementedError

    @property
    def n(self) -> int:
        raise NotImplementedError

    def standardize(self, image: np.ndarray):
        return image
