import re
from typing import Optional, Any

import numpy as np

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.metrics.AbstractMetric import AbstractMetric


class MAE(BaseClass, AbstractMetric):
    def __init__(self, name: str):
        super().__init__(name)

    @staticmethod
    def parser(name: str) -> Optional[Any]:
        if re.match("^mae$", name):
            return name
        return None

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(np.mean(np.abs(pred_batch - true_batch), axis=1))
