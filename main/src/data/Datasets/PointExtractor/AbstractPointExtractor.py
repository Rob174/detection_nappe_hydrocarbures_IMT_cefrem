from abc import ABC, abstractmethod

from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories


class AbstractPointExtractor(ABC):
    """Responsible to extract the points from the dict"""
    @abstractmethod
    def extract(self,data):
        """Responsible to extract the points from the dict"""

        