"""Contains enumerations EnumLabelModifier, EnumClassPatchAdder"""

from enum import Enum


class EnumLabelModifier(str, Enum):
    NoLabelModifier = "nolabelmodifier"
    """class other,seep,spill label"""
    LabelModifier1 = "labelmodifier1"
    """classes constructed to allow to choose classes to use"""
    LabelModifier2 = "labelmodifier2"
    """1 class constructed to allow to search if a combination of classes is present"""


class EnumClassPatchAdder(str, Enum):
    OtherClassPatchAdder = "other_class_patch_adder"
    NoClassPatchAdder = "no_class_patch_adder"
