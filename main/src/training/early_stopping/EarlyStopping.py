"""Checks the precedent metrics and determine if we have to stop the training process based on a min_delta per epoch and a patience
If the metric does not diminish from more than min_delta during more than patience, we stop the training process"""

from enum import Enum

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.early_stopping.AbstractEarlyStopping import AbstractEarlyStopping
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager


class EarlyStopping(BaseClass, AbstractEarlyStopping):
    """Checks the precedent metrics and determine if we have to stop the training process based on a min_delta per epoch and a patience
        If the metric does not diminish from more than min_delta during more than patience, we stop the training process
    """
    def __init__(self, metric: AbstractMetricManager, name_metric_chosen: Enum, patience: int = 10, min_delta: float = 0.1):
        super(EarlyStopping, self).__init__(metric, name_metric_chosen)
        self.attr_name = self.__class__.__name__
        self.metric: AbstractMetricManager = metric
        self.attr_name_metric_chosen: Enum = name_metric_chosen
        self.attr_patience_threshold = patience
        self.attr_min_delta = abs(min_delta)

        self.last_epoch_metric_value = -1
        self.status = 0

    def stop_training(self) -> bool:
        """Checks the precedent metrics and determine if we have to stop the training process
        If the metric does not diminish from more than min_delta during more than patience, we stop the training process
        """
        metric_value = self.metric.get_last_metric(self.attr_name_metric_chosen)
        if self.last_epoch_metric_value != -1 and (self.last_epoch_metric_value-metric_value) <= self.attr_min_delta:
            self.status += 1
        else:
            self.status = 0
        self.last_epoch_metric_value = metric_value
        if self.status > self.attr_patience_threshold:
            return True
        else:
            return False
