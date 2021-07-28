"""Checks the precedent metrics and determine if we have to stop the training process based on a min_delta per epoch and a patience
If the metric does not diminish from more than min_delta during more than patience, we stop the training process"""

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.early_stopping.AbstractEarlyStopping import AbstractEarlyStopping

from main.src.training.metrics.AbstractMetric import AbstractMetric


class EarlyStopping(BaseClass, AbstractEarlyStopping):
    """Checks the precedent metrics and determine if we have to stop the training process based on a min_delta per epoch and a patience
        If the metric does not diminish from more than min_delta during more than patience, we stop the training process
    """

    def __init__(self, metric: AbstractMetric, patience: int = 10, min_delta: float = 0.1):
        super(EarlyStopping, self).__init__(metric)
        self.metric: AbstractMetric = metric
        self.attr_patience_threshold = patience
        self.attr_min_delta = abs(min_delta)

        self.last_epoch_metric_value = -1
        self.status_patience = 0
        self.status_training = True

    def stop_training(self):
        """Checks the precedent metrics and determine if we have to stop the training process
        If the metric does not diminish from more than min_delta during more than patience, we stop the training process
        """
        metric_value = self.metric.get_last_value()
        if self.last_epoch_metric_value != -1 and (self.last_epoch_metric_value - metric_value) <= self.attr_min_delta:
            self.status_patience += 1
        else:
            self.status_patience = 0
        self.last_epoch_metric_value = metric_value
        if self.status_patience > self.attr_patience_threshold:
            self.status_training = False
