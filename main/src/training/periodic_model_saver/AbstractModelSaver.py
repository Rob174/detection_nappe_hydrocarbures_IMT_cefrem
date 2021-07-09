from abc import ABC, abstractmethod

import torch

from main.FolderInfos import FolderInfos
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.metrics.metrics_factory import MetricsFactory


class AbstractModelSaver(BaseClass, ABC):
    def __init__(self, loss: LossFactory, metrics: MetricsFactory):
        pass

    @abstractmethod
    def save_model_if_required(self, model: ModelFactory, epoch, iteration):
        pass

    def save_model(self, model: ModelFactory, epoch, iteration):
        torch.save(model.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{iteration}.pt")
