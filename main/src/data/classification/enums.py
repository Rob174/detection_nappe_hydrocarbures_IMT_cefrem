from enum import Enum


class EnumClassificationDataset(Enum):
    ClassificationPatch = "classificationpatch"
    """dataset of classification on patches with original classes"""
    ClassificationPatch1 = "classificationpatch1"
    """dataset of classification on patches with less classes than the original dataset"""
    ClassificationPatch2 = "classificationpatch2"
    """dataset of classification on patches merging specified classes together to predict if there is something or not"""