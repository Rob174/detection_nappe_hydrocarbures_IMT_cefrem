"""Model saver saving model at a fixed step rate"""

from enum import Enum

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager
from main.src.training.periodic_model_saver.AbstractModelSaver import AbstractModelSaver


class ModelSaver1(BaseClass, AbstractModelSaver):
    """Model saver saving model at a fixed step rate"""
    def __init__(self, metric_used: AbstractMetricManager, metric_name: Enum):
        super(ModelSaver1, self).__init__(metric_used,metric_name)
        self.attr_name = self.__class__.__name__
        self.metric_used = metric_used
        self.attr_metric_name = metric_name
        self.attr_step = 100
        self.it_call = 0

    def save_model_if_required(self, model, epoch, iteration):
        """Check if we have tp save the model based on the informations specified in the constructor (directly written in the code for the moment)"""
        if self.metric_used.get_last_metric(self.attr_metric_name) and self.it_call % self.attr_step == 0:
            self.save_model(model, epoch, iteration)
        self.it_call += 1
