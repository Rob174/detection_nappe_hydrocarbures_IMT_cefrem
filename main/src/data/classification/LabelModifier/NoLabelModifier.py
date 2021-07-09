import numpy as np

from main.src.data.classification.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.data.enums import EnumClasses


class NoLabelModifier(AbstractLabelModifier):
    def __init__(self, *args, **kwargs):
        super(NoLabelModifier, self).__init__()
        self.attr_classes_to_use = [EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill]
        pass

    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        return annotation
