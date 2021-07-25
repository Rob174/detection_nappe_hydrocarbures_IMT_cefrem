"""Base class to build a model saver"""

from abc import ABC, abstractmethod
from enum import Enum

import torch

from main.FolderInfos import FolderInfos
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager
from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.metrics.metrics_factory import MetricsFactory


class AbstractModelSaver( ABC):
    """Base class to build a model saver"""
    def __init__(self):

        pass

    @abstractmethod
    def save_model_if_required(self, model: ModelFactory, epoch, iteration):
        """Check if we have tp save the model based on the informations specified in the constructor (directly written in the code for the moment)"""
        pass

    def save_model(self, model: ModelFactory, epoch, iteration):
        """Saves the model in a pt file with informations about the status of the training"""
        torch.save(model.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{iteration}.pt")
