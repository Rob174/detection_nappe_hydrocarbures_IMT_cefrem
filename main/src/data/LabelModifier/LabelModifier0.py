"""Modify the source annotation image (2d array). Allow to transform the 2d array into a vector of probabilities of presence of each class"""

import numpy as np

from main.src.data.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.param_savers.BaseClass import BaseClass


class LabelModifier0(AbstractLabelModifier, BaseClass):
    """Modify the source annotation image (2d array). Allow to transform the 2d array into a vector of probabilities of presence of each class

    Args:
        class_mapping: TwoWayDict, map index of class to its name
    """

    def __init__(self, class_mapping: TwoWayDict):
        super().__init__()

        self.attr_name = self.__class__.__name__
        self.attr_class_mapping = class_mapping
        self.attr_threshold = 16

    def get_final_class_mapping(self):
        return self.attr_class_mapping

    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        """Creates the Generators label based on the annotation patch image

        Args:
            annotation: np.ndarray 2d containing for each pixel the class of this pixel

        Returns: the Generators label

        """

        classification_label = np.zeros((len(self.attr_class_mapping),), dtype=np.float32)  # 0 ns
        values_present,eff = np.unique(classification_label,return_counts=True)
        for value in self.attr_class_mapping.keys(Way.ORIGINAL_WAY):
            # for each class of the original attr_dataset, we put a probability of presence of one if the class is in the patch
            value = int(value)

            #  if the class is in the patch
            if value in values_present and eff[values_present == value] > self.attr_threshold:
                classification_label[value] = 1.
        self.initial_label = classification_label
        return classification_label
