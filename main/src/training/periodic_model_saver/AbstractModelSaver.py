"""Base class to build a model saver"""

from abc import ABC, abstractmethod

import torch

from main.FolderInfos import FolderInfos
from main.src.models.ModelFactory import ModelFactory
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.metrics.AbstractMetric import AbstractMetric


class AbstractModelSaver(AbstractCallback):
    """Base class to build a model saver"""
    def __init__(self, metric_used: AbstractMetric,model: ModelFactory):
        super(AbstractModelSaver, self).__init__()
        self.model = model
        self.metric_used = metric_used
        self.attr_metric_name = metric_used.name

    def on_valid_end(self, prediction_batch, true_batch):
        self.save_model_if_required(self.model)
    def on_epoch_end(self):
        self.save_model_if_required(self.model)

    @abstractmethod
    def save_model_if_required(self, model: ModelFactory):
        """Check if we have tp save the model based on the informations specified in the constructor (directly written in the code for the moment)"""
        pass

    def save_model(self, model: ModelFactory, epoch, iteration):
        """Saves the model in a pt file with informations about the status_patience of the training"""
        torch.save(model.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{iteration}.pt")
