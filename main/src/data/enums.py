from enum import Enum


class EnumUsage(Enum,str):
    Classification = "classification"
    """dataset to classify patches"""
    Segmentation = "segmentation"
    """dataset to segment an image"""

class EnumClasses(Enum,str):
    Other = "other"
    Seep = "seep"
    Spill = "spill"