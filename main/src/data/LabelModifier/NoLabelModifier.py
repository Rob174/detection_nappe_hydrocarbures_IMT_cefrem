"""Class use to have the same interface. Makes no modifications"""

import numpy as np

from main.src.data.TwoWayDict import TwoWayDict
from main.src.data.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.enums import EnumClasses
from main.src.param_savers.BaseClass import BaseClass


class NoLabelModifier(AbstractLabelModifier,BaseClass):
    """Class use to have the same interface. Makes no modifications"""

    def __init__(self, original_class_mapping: TwoWayDict, *args, **kwargs):
        super(NoLabelModifier, self).__init__()
        self.attr_classes_to_use = [EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill]
        self.attr_class_mapping = original_class_mapping

    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        """Makes no modifications to the annotation"""
        return annotation
