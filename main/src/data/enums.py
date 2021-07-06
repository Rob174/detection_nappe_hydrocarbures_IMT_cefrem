from enum import Enum


class EnumUsage(str,Enum):
    Classification = "classification"
    """dataset to classify patches"""
    Segmentation = "segmentation"
    """dataset to segment an image"""

class EnumClasses(str,Enum):
    Other = "other"
    Seep = "seep"
    Spill = "spill"