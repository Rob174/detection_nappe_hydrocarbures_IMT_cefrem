"""
Module regrouping classes to modify the initial annotation, mainly for the moment to create a vector of probabilities

Usage:
>>> from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
>>> from main.src.data.TwoWayDict import TwoWayDict
>>> from main.src.enums import EnumClasses
>>> modifier = LabelModifier1(original_class_mapping=TwoWayDict({
...                             0: "other",
...                             1: "seep",
...                             2: "spill",
...                         }),
...                           classes_to_use=(EnumClasses.Seep,EnumClasses.Spill))
>>> annotation_2d = ...
>>> modifier.make_classification_label(annotation_2d)
np.ndarray([proba_seep,proba_spill])
"""



