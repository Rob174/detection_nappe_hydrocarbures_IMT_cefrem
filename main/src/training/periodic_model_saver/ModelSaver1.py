"""Model saver saving model at a fixed step rate"""

from enum import Enum

from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.IterationManager import IterationManager
from main.src.training.metrics.AbstractMetric import AbstractMetric
from main.src.training.periodic_model_saver.AbstractModelSaver import AbstractModelSaver


class ModelSaver1(BaseClass, AbstractModelSaver):
    """Model saver saving model at a fixed step rate"""
    def __init__(self, metric_used: AbstractMetric, model: ModelFactory,iteration_manager: IterationManager):
        super(ModelSaver1, self).__init__(metric_used,model)
        self.attr_step = 100
        self.it_call = 0
        self.iteration_manager = iteration_manager

    def save_model_if_required(self, model):
        """Check if we have tp save the model based on the informations specified in the constructor (directly written in the code for the moment)"""
        if self.metric_used.get_last_value() and self.it_call % self.attr_step == 0:
            self.save_model(model, self.iteration_manager.attr_last_epoch, self.iteration_manager.attr_it_tr)
        self.it_call += 1
