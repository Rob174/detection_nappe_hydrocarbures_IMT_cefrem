from abc import ABC
from enum import Enum


class AbstractMetricManager(ABC):
    def get_last_metric(self, name: Enum) -> float:
        pass
