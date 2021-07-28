import re
from typing import Optional, Any

import numpy as np

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.metrics.AbstractMetric import AbstractMetric


class AccuracyClassification(BaseClass, AbstractMetric):
    def __init__(self, name: str, precision: float):
        super().__init__(name)
        self.attr_precision = precision

    @staticmethod
    def parser(name: str) -> Optional[Any]:
        if re.match("^accuracy_classification-[0-9]\\.[0-9]+$", name):
            precision = float(re.sub("^accuracy_classification-([0-9]\\.[0-9]+)$", "\\1", name))
            return name, precision
        return None

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(np.mean((np.abs(pred_batch - true_batch) < self.attr_precision).astype(np.float32), axis=1))
