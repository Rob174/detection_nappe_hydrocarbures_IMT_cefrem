from enum import Enum


class EnumUsage(Enum):
    Classification = "classification"
    """dataset to classify patches"""
    Segmentation = "segmentation"
    """dataset to segment an image"""

class EnumClasses(Enum):
    Other = "other"
    Seep = "seep"
    Spill = "spill"