"""Contains enumerations EnumPatchAlgorithm and EnumPatchExcludePolicy"""

from enum import Enum


class EnumPatchAlgorithm(str, Enum):
    FixedPx = "fixed_px"
    """create the PatchCreator0 class which generate fixed px size patches. See [patch_creator0](./patch_creator0.html)"""


class EnumPatchExcludePolicy(str, Enum):
    MarginMoreThan = "marginmorethan"
    """Choose the option to exclude patches that have more than ... px at 0 in order to exclude patches with margins"""
