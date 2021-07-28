from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any

import main.src.training.metrics.AbstractMetric as abstractmetric
from main.src.enums import EnumDataset
from main.src.training.AbstractCallback import AbstractCallback


class AbstractMetric(abstractmetric.AbstractMetric, AbstractCallback):
    def __init__(self, name: str, *args, **kwargs):
        self.attr_name = name
        self.attr_values: Dict[str, List[float]] = {EnumDataset.Train: [], EnumDataset.Valid: []}

    @staticmethod
    def parser(name: str) -> Optional[Any]:
        """Method that tells if the name correspond to this class by providing arguments if it matches
        Put the original place in this case"""
        return None

    @abstractmethod
    def npy_compute(self, true_batch, pred_batch):
        """Compute the metric function on the true and predicted batch using numpy functions"""

    def on_train_end(self, true_batch, pred_batch):
        """Compute and save the metric"""
        current = self.npy_compute(true_batch, pred_batch)
        self.attr_values[EnumDataset.Train].append(current)

    def on_valid_start(self, true_batch, pred_batch):
        """Evaluates the model on valid batches"""
        current = self.npy_compute(true_batch, pred_batch)
        self.attr_values[EnumDataset.Valid].append(current)

    @property
    def values(self):
        return self.attr_values

    @property
    def name(self):
        return self.attr_name
