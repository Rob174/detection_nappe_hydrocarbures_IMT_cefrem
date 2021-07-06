from typing import Tuple

import numpy as np

from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.data.enums import EnumClasses


class LabelModifier2(AbstractLabelModifier):
    """Modify the source label provided by the source class inheriting from AbstractClassificationDataset

    Args:
        original_class_mapping: TwoWayDict mapping other, seep, spill... categories to their ids in the original 1d classification label
        classes_to_use: Tuple[EnumClasses], indicates the classes to use in the final classification label
    """

    def __init__(self, original_class_mapping: TwoWayDict,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
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
        self.attr_class_mapping_merged = tmp_mapping
        self.attr_global_name = "dataset"

    def make_classification_label(self, annotation: np.ndarray):
        """Creates the classification label based on the annotation patch image

        Merge specified classes together

        Args:
            annotation: np.ndarray 1d containing the probability that the patch contain the classes as specified in NoLabelModifier make_classification_label method

        Returns:
            annotation_modified: the classification label modified

        """
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        annotation_modified = np.zeros((1,))
        src_indexes = list(map(int, self.attr_class_mapping_merged.keys(Way.ORIGINAL_WAY)[0].split("|")))
        # Merging selected classes together with the max
        for src_index in src_indexes:
            annotation_modified[0] = max(annotation_modified[0], annotation[src_index])
        return annotation_modified
