"""Use the color code provided with inside the dict"""

from main.src.data.Datasets.ColorExtractor.AbstractColorExtractor import AbstractColorExtractor
from main.src.data.TwoWayDict import TwoWayDict
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories


class DicoColorExtractor(AbstractColorExtractor):
    """Use the color code provided with inside the dict"""
    def __init__(self,mapping: TwoWayDict):
        super(DicoColorExtractor, self).__init__(mapping)

    def extract(self, dico: dict) -> str:
        """Return the color code found in the dict"""
        label: int = self.attr_mapping[dico[EnumShapeCategories.Label]]
        color_code = "#" + f"{label:02x}" * 3  # conversion to hexadecimal color (#FFFFFF for white for instance)
        return color_code