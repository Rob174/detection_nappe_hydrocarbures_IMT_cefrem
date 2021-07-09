from enum import Enum

import torch

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager
from main.src.training.periodic_model_saver.AbstractModelSaver import AbstractModelSaver


class ModelSaver1(BaseClass,AbstractModelSaver):
    def __init__(self,metric_used: AbstractMetricManager,metric_name:Enum):
        self.attr_name = self.__class__.__name__
        self.metric_used = metric_used
        self.attr_metric_name = metric_name
        self.attr_step = 100
    def save_model_if_required(self,model,epoch,iteration):
        if self.metric_used.get_last_metric(self.attr_metric_name) and self.it_call % self.attr_step == 0:
            self.save_model(model,epoch,iteration)