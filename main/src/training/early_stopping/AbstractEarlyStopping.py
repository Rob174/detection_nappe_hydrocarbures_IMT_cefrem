from abc import ABC,abstractmethod

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager


class AbstractEarlyStopping(ABC,BaseClass):
    def __init__(self,metric: AbstractMetricManager,name_metric_chosen):
        self.epochs_metric_values = []
        self.attr_metric: AbstractMetricManager = metric
        self.attr_name_metric_chosen = name_metric_chosen

    @abstractmethod
    def stop_training(self) -> bool:
        pass