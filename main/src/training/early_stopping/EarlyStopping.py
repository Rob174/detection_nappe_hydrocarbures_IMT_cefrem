from enum import Enum

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.early_stopping.AbstractEarlyStopping import AbstractEarlyStopping
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager


class EarlyStopping(BaseClass, AbstractEarlyStopping):
    def __init__(self, metric: AbstractMetricManager, name_metric_chosen: Enum, patience: int = 3):
        super(EarlyStopping, self).__init__(metric, name_metric_chosen)
        self.attr_name = self.__class__.__name__
        self.metric: AbstractMetricManager = metric
        self.attr_name_metric_chosen: Enum = name_metric_chosen
        self.attr_patience_threshold = patience

        self.last_epoch_metric_value = -1
        self.status = 0

    def stop_training(self) -> bool:
        metric_value = self.metric.get_last_metric(self.attr_name_metric_chosen)
        if self.last_epoch_metric_value != -1 and self.last_epoch_metric_value <= metric_value:
            self.status += 1
        else:
            self.status = 0
        if self.status > self.attr_patience_threshold:
            return True
        else:
            return False
