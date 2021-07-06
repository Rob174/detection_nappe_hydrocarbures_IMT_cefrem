from enum import Enum


class EnumAugmenter(Enum):
    Augmenter0 = "augmenter0"
    """A step by step augmenter. See Augmenter0"""
    Augmenter1 = "augmenter1"
    """A one step augmenter. See Augmenter1"""
    NoAugmenter = "noaugmenter"
    """No augmentation applied"""