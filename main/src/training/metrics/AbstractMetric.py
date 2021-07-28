from abc import ABC

from main.src.enums import EnumDataset


class AbstractMetric(ABC):
    @property
    def values(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def get_last_value(self) -> float:
        """Get last values of the metric asked"""
        return self.values[EnumDataset.Valid][-1]
