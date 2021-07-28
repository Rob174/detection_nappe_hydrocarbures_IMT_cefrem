"""Abstract class to specify the expected interface for a callback (to be developped in the future)"""

from abc import ABC, abstractmethod

import torch

import numpy as np


class AbstractCallback(ABC):
    """Abstract class to specify the expected interface for a callback (to be developped in the future)"""


    def on_train_start(self, prediction_batch: torch.Tensor, true_batch: torch.Tensor):
        """Called for loss calculation"""
    def on_train_end(self, prediction_batch: np.ndarray, true_batch: np.ndarray):
        """For free usage after loss calculation"""

    def on_valid_start(self, prediction_batch, true_batch):
        """Called for all metrics calculations """

    def on_valid_end(self, prediction_batch, true_batch):
        """Called after a valid batch has been tested on the model and metrics has been calculated"""

    def on_epoch_end(self):
        """called when an epoch ends"""

    def on_epoch_start(self, epoch):
        """called when an epoch ends"""
    def on_end(self):
        """called when an epoch ends"""
