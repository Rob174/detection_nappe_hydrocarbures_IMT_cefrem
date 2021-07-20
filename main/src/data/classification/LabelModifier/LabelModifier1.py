"""Modify the source label provided by the source class inheriting. Allow to use less classes than other, seep, spill
and to change their order."""

from typing import Tuple

import numpy as np

from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.LabelModifier.AbstractLabelModifier import AbstractLabelModifier
from main.src.data.classification.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.enums import EnumClasses


class LabelModifier1(AbstractLabelModifier):
    """Modify the source label provided by the source class inheriting. Allow to use less classes than other, seep, spill
    and to change their order.

    Args:
        classes_to_use: Tuple[EnumClasses], indicates the classes to use in the final classification label
    """

    def __init__(self, original_class_mapping: TwoWayDict,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
        super().__init__()
        self.label_modifier0 = LabelModifier0(class_mapping=original_class_mapping)
        self.attr_name = self.__class__.__name__
        tmp_mapping = TwoWayDict({})
        # modifying the class mappings according to attributes provided
        self.attr_classes_to_use = classes_to_use
        for i, name in enumerate(classes_to_use):
            tmp_mapping[original_class_mapping[name.value], Way.ORIGINAL_WAY] = name.value, i
        self.attr_class_mapping = tmp_mapping
        self.initial_label = None
    def get_final_class_mapping(self):
        dico = TwoWayDict({})
        for original_id,(name,dest_id) in self.attr_class_mapping.items(dico_chosen=Way.ORIGINAL_WAY):
            dico[dest_id, Way.ORIGINAL_WAY] = name
        return dico
    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        """Creates the classification label based on the annotation patch image

        Args:
            annotation:  np.ndarray 2d containing for each pixel the class of this pixel

        Returns:
            annotation_modified: the classification label modified

        """
        annotation = self.label_modifier0.make_classification_label(annotation)
        # Create a classification label with less classes
        annotation_modified = np.zeros((len(self.attr_class_mapping.keys()),))
        # {index_src_0:(class0,index_dest0),index_src_1:(class1,index_dest1),....}
        for src_index, (_, dest_index) in self.attr_class_mapping.items(Way.ORIGINAL_WAY):
            annotation_modified[dest_index] = annotation[src_index]
        return annotation_modified
