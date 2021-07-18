"""
Augmenters classes which manage and apply augmentations.

There are two variants of the augmenters applying respectively the augmentation step after step for Augmenter0
and all at once for Augmenter1. Augmenter1 is an optimized version of Augmenter0. That is why it provides a different interface
than Augmenter0 (cf respective docs)

Responsabilities:
- Create the augmentations
- Apply the augmentations
"""
