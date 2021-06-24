"""
The goal of this module is to manage and apply augmentations.

As we will apply the same types of augmentations to all of the images,
we have a first object Augmenter... in the package Augmenters
which we first neeed to initialize.
It allows us to specify the augmentations to use and in which order to apply them (cf doc)

>>> augmenter = Augmenter0(allowed_transformations="mirrors,rotations")

Then we can use this object to apply random transformations
to input ⚠️ images (not batch of images).

All of the transformations are randomly chosen and can change between epochs

>>> image = ...
>>> annotation = ...
>>> array,annotation = augmenter.transform(image,annotation)

To allow to keep a constant interface, we made a NoAugmenter class which performs
no augmentation and directly returns the input
"""