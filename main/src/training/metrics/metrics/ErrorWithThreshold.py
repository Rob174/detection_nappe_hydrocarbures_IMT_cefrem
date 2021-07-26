from typing import Tuple, Optional, Any

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.metrics.AbstractMetric import AbstractMetric

import numpy as np
import re


class ErrorWithThreshold(BaseClass,AbstractMetric):
    def __init__(self, name: str, threshold: float):
        super().__init__(name)
        self.attr_threshold = threshold
    @staticmethod
    def parser(name:str) -> Optional[Any]:
        if re.match("^error_threshold-[0-9]\\.[0-9]+$", name):
            threshold = float(re.sub("^error_threshold-([0-9]\\.[0-9]+)$", "\\1", name))
            return name,threshold
        return None
    def npy_compute(self,true_batch,pred_batch):
        new_pred = np.copy(pred_batch)
        new_true = np.copy(true_batch)
        new_pred[pred_batch > self.attr_threshold] = 1.
        new_pred[pred_batch <= self.attr_threshold] = 0.
        new_true[true_batch > self.attr_threshold] = 1.
        new_true[true_batch <= self.attr_threshold] = 0.
        return np.sum(np.abs(new_pred-new_true))

