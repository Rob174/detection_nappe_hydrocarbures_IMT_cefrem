from abc import ABC, abstractmethod

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager


class AbstractEarlyStopping(ABC):
    """Base class to build an early stopping algorithm

    Args:
        metric: AbstractMetricManager, object storing the metric used for the early stopping
        name_metric_chosen: str name of the metric chosen inside this metric manager
    """
    def __init__(self, metric: AbstractMetricManager, name_metric_chosen: str):
        self.epochs_metric_values = []
        self.metric: AbstractMetricManager = metric
        self.attr_name_metric_chosen = name_metric_chosen

    @abstractmethod
    def stop_training(self) -> bool:
        """Checks the precedent metrics and determine if we have to stop the training process"""
        pass
