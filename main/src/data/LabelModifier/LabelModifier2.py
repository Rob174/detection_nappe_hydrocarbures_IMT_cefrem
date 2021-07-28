"""Merge multiple labels the classes to use and indicate the presence or absence of one of these classes"""

from typing import Tuple

import numpy as np

from main.src.data.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.enums import EnumClasses
from main.src.param_savers.BaseClass import BaseClass


class LabelModifier2(AbstractLabelModifier, BaseClass):
    """Modify the source label provided by the LabelModifier0 (done internally).
    Merge multiple labels the classes to use and indicate the presence or absence of one of these classes

    Args:
        original_class_mapping: TwoWayDict mapping other, seep, spill... categories to their ids in the original 1d Generators label
        classes_to_use: Tuple[EnumClasses], indicates the classes to use in the final Generators label
    """

    def __init__(self, original_class_mapping: TwoWayDict,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
        super().__init__()
        self.label_modifier0 = LabelModifier0(class_mapping=original_class_mapping)
        self.attr_name = self.__class__.__name__
        tmp_mapping = TwoWayDict({})
        self.attr_classes_to_use = classes_to_use
        lkey = []
        lvalue = []
        lname = []
        # merge classes in the dict
        for i, name in enumerate(classes_to_use):
            lkey.append(str(original_class_mapping[name]))
            lvalue.append(str(i))
            lname.append(name.value)
        tmp_mapping["|".join(lkey), Way.ORIGINAL_WAY] = "|".join(lname), "|".join(lvalue)
        self.attr_class_mapping = tmp_mapping
        self.attr_global_name = "attr_dataset"

    def make_classification_label(self, annotation: np.ndarray):
        """Creates the Generators label based on the annotation patch image

        Merge specified classes together

        Args:
            annotation: np.ndarray 2d containing for each pixel the class of this pixel

        Returns:
            annotation_modified: the Generators label modified

        """
        self.initial_label = self.label_modifier0.make_classification_label(annotation)
        annotation = self.initial_label
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        annotation_modified = np.zeros((1,))
        src_indexes = list(map(int, self.attr_class_mapping.keys(Way.ORIGINAL_WAY)[0].split("|")))
        # Merging selected classes together with the max
        for src_index in src_indexes:
            annotation_modified[0] = max(annotation_modified[0], annotation[src_index])
        return annotation_modified
