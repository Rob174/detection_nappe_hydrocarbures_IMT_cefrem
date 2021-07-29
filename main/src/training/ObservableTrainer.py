import numpy as np
import torch
from typing import List

from main.src.training.AbstractCallback import AbstractCallback


class ObservableTrainer:
    def __init__(self,callbacks: List[AbstractCallback]):
        self.attr_callbacks: List[AbstractCallback] = callbacks
    def on_train_start(self,pred_batch: torch.Tensor,true_batch: torch.Tensor):
        """Called when at each train step. To be uniquely used by AbstractLoss object"""
        for callback in self.attr_callbacks:
            callback.on_train_start(pred_batch,true_batch)

    def on_train_end(self,pred_batch: np.ndarray,true_batch: np.ndarray):
        """Called when at each train step after loss computation"""
        for callback in self.attr_callbacks:
            callback.on_train_end(pred_batch,true_batch)

    def on_valid_start(self,pred_batch,true_batch):
        """Called when at each valid step. To be uniquely used by AbstractMetric object"""
        for callback in self.attr_callbacks:
            callback.on_valid_start(pred_batch,true_batch)
    def on_valid_end(self,pred_batch,true_batch):
        """Called at each valid step after metrics calculation"""
        for callback in self.attr_callbacks:
            callback.on_valid_end(pred_batch,true_batch)
    def on_epoch_start(self,epoch):
        for callback in self.attr_callbacks:
            callback.on_epoch_start(epoch)

    def on_epoch_end(self):
        for callback in self.attr_callbacks:
            callback.on_epoch_end()

    def on_end(self):
        for callback in self.attr_callbacks:
            callback.on_end()
    def on_start(self):
        for callback in self.attr_callbacks:
            callback.on_start()
