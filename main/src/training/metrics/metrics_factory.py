from typing import List

from main.src.training.metrics.metrics.AbstractMetric import AbstractMetric
from main.src.training.metrics.metrics.AccuracyClassification import AccuracyClassification
from main.src.training.metrics.metrics.ErrorWithThreshold import ErrorWithThreshold
from main.src.training.metrics.metrics.MAE import MAE


class MetricsFactory:
    """Class managing metrics

    Args:
        metrics_names: iterable of metrics as str. Currently supported metrics:

    - "accuracy_classification-[0-9]\.[0-9]+": mean of the the mean number of times where the difference between prediction and reference are less than the threashold provided after accuracy_classification-
    - "error_threshold-[0-9]\.[0-9]+
    - "mae": mean average error

    Example:
        >>> MetricsFactory.create("accuracy_classification-0.9",
        ...                "mae","accuracy_classification-0.95")
        ...                # this is the way to provide the iterable of metrics
    """

    @staticmethod
    def create(*metrics_names: str) -> List[AbstractMetric]:
        available_metrics = [
            AccuracyClassification,
            ErrorWithThreshold,
            MAE
        ]
        list_metrics = []
        for metric in metrics_names:
            metric = metric.lower()
            for Metric_class in available_metrics:
                args = Metric_class.parser(metric)
                if args is not None:
                    list_metrics.append(Metric_class(*args))
        return list_metrics
