from enum import Enum


class EnumUsage(str,Enum):
    Classification = "classification"
    """attr_dataset to classify patches"""
    Segmentation = "segmentation"
    """attr_dataset to segment an image"""

class EnumClasses(str,Enum):
    Other = "other"
    Seep = "seep"
    Spill = "spill"