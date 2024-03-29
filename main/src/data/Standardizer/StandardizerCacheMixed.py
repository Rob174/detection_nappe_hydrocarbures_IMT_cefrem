"""Standardizer to apply when we use the OtherClassPatchAdder class"""

import numpy as np

from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.data.Standardizer.StandardizerCacheOther import StandardizerCacheOther
from main.src.data.Standardizer.StandardizerCacheSeepSpill import StandardizerCacheSeepSpill
from main.src.param_savers.BaseClass import BaseClass


class StandardizerCacheMixed(BaseClass, AbstractStandardizer):
    """Standardizer to apply when we use the OtherClassPatchAdder class

    Args:
        interval: interval between two patches with only the other class
    """

    def __init__(self, interval):
        super().__init__()
        self.ratio = interval
        self.stats_seep_spill = StandardizerCacheSeepSpill()
        self.stats_other = StandardizerCacheOther()

    @property
    def mean(self):
        return ((self.stats_seep_spill.mean / self.stats_other.n) + (
                    self.stats_other.mean / (self.ratio * self.stats_seep_spill.n))) / \
               self.n * (self.stats_seep_spill.n * self.stats_other.n)

    @property
    def std(self):
        return ((self.stats_seep_spill.std / self.stats_other.n) + (
                    self.stats_other.std / (self.ratio * self.stats_seep_spill.n))) / \
               self.n * (self.stats_seep_spill.n * self.stats_other.n)

    @property
    def n(self) -> int:
        return self.stats_seep_spill.n * self.ratio + self.stats_other.n

    def standardize(self, image: np.ndarray):
        return (image - self.mean) / self.std