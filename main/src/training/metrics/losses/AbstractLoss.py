from abc import ABC, abstractmethod
from typing import Dict, List

import torch

from main.src.enums import EnumDataset
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.metrics.AbstractMetric import AbstractMetric
from main.src.training.optimizers.optimizers.AbstractOptimizer import AbstractOptimizer


class AbstractLoss(ABC, AbstractMetric, AbstractCallback):
    """Base class to represent a loss"""

    def __init__(self, optimizer: AbstractOptimizer):
        self.attr_optimizer = optimizer
        self.attr_values: Dict[str, List[float]] = {EnumDataset.Train: [], EnumDataset.Valid: []}

    @abstractmethod
    def torch_compute(self, true_batch, pred_batch):
        """Compute the loss function on the true and predicted batch using pytorch functions"""

    @abstractmethod
    def npy_compute(self, true_batch, pred_batch):
        """Compute the same loss function on the true and predicted batch using numpy functions"""

    def on_train_start(self, prediction_batch: torch.Tensor, true_batch: torch.Tensor):
        """Make forward and backward propagation and saves the loss"""
        loss = self.torch_compute(true_batch, prediction_batch)
        loss.backward()
        self.attr_optimizer.step()

    def on_train_end(self, prediction_batch, true_batch):
        current_loss = self.npy_compute(true_batch, prediction_batch)
        self.attr_values[EnumDataset.Train].append(current_loss)

    def on_valid_start(self, prediction_batch, true_batch):
        """Evaluates the model on valid batches"""
        current_loss = self.npy_compute(true_batch, prediction_batch)
        self.attr_values[EnumDataset.Valid].append(current_loss)

    def zeros_grad(self):
        self.attr_optimizer.zero_grad()

    @property
    def values(self):
        return self.attr_values

    @property
    def name(self):
        return self.attr_name

    def get_last_tr_loss(self):
        return self.attr_values[EnumDataset.Train][-1]
