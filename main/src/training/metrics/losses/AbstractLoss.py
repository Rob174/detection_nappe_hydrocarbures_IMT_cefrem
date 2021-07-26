from abc import ABC,abstractmethod

from typing import Dict, List

from main.src.enums import EnumDataset
from main.src.training.metrics.AbstractMetric import AbstractMetric


class AbstractLoss(ABC,AbstractMetric):
    """Base class to represent a loss"""
    def __init__(self,optimizer: AbstractOptimizer):
        self.attr_optimizer = optimizer
        self.attr_values: Dict[str, List[float]] = {EnumDataset.Train: [], EnumDataset.Valid: []}
    @abstractmethod
    def torch_compute(self,true_batch,pred_batch):
        """Compute the loss function on the true and predicted batch using pytorch functions"""
    @abstractmethod
    def npy_compute(self,true_batch,pred_batch):
        """Compute the same loss function on the true and predicted batch using numpy functions"""

    def on_train_start(self,true_batch,pred_batch):
        """Make forward and backward propagation and saves the loss"""
        loss = self.torch_compute(true_batch,pred_batch)
        loss.backward()
        self.attr_optimizer().step()
        current_loss = self.npy_compute(true_batch,pred_batch)
        self.attr_values[EnumDataset.Train].append(current_loss)
        return current_loss
    def on_train_end(self,true_batch,pred_batch):
        pass
    def on_valid_start(self,true_batch,pred_batch):
        """Evaluates the model on valid batches"""
        current_loss = self.npy_compute(true_batch,pred_batch)
        self.attr_values[EnumDataset.Valid].append(current_loss)
        return current_loss
    def on_valid_end(self,true_batch,pred_batch):
        pass

    @property
    def values(self):
        return self.attr_values
    @property
    def name(self):
        return self.attr_name