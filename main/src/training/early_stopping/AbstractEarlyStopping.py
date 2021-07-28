from abc import ABC, abstractmethod

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.metrics.AbstractMetric import AbstractMetric


class AbstractEarlyStopping(ABC,AbstractCallback):
    """Base class to build an early stopping algorithm

    Args:
        metric: AbstractMetricManager, object storing the metric used for the early stopping
        name_metric_chosen: str name of the metric chosen inside this metric manager
    """
    def __init__(self, metric: AbstractMetric):
        self.epochs_metric_values = []
        self.metric: AbstractMetric = metric
        self.attr_name_metric = metric.name
    def on_epoch_end(self):
        self.stop_training()
    @abstractmethod
    def stop_training(self):
        """Checks the precedent metrics and determine if we have to stop the training process"""
        pass
