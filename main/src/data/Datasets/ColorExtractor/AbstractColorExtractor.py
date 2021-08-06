"""Base class for a color extractor which has the responsability to provide the color to use for each polygon provided (represented as dict)"""
from abc import ABC, abstractmethod

from main.src.data.TwoWayDict import TwoWayDict


class AbstractColorExtractor(ABC):
    """Base class for a color extractor which has the responsability to provide the color to use for each polygon provided (represented as dict)"""
    def __init__(self,mapping: TwoWayDict):
        self.attr_mapping = mapping

    @abstractmethod
    def extract(self,dico: dict) -> str:
        """Returns the color to use on the final image in the hexadecimal format"""