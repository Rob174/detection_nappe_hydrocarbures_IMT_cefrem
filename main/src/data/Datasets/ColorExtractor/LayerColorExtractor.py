"""Put a constant color for these points"""
from main.src.data.Datasets.ColorExtractor.AbstractColorExtractor import AbstractColorExtractor
from main.src.data.TwoWayDict import TwoWayDict


class LayerColorExtractor(AbstractColorExtractor):
    """Put a constant color for these points"""
    def __init__(self,mapping: TwoWayDict, fixed_color_code: str):
        super(LayerColorExtractor, self).__init__(mapping)
        self.attr_fixed_color_code = fixed_color_code

    def extract(self, dico: dict) -> str:
        """Returns the constant color code chosen"""
        return self.attr_fixed_color_code
        