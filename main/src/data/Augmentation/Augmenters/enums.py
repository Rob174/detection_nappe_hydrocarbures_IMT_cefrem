"""Contains enumeration EnumAugmenter"""

from enum import Enum


class EnumAugmenter(str, Enum):
    Augmenter0 = "augmenter0"
    """A step by step augmenter. See [Augmenter0](./Augmenter0.html)"""
    Augmenter1 = "augmenter1"
    """A one step augmenter. See [Augmenter1](./Augmenter1.html)"""
    NoAugmenter = "noaugmenter"
    """No augmentation applied"""
