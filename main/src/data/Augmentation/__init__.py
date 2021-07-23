"""
The goal of this module is to manage and apply augmentations.

As we will apply the same types of augmentations to all of the images,
we have a first object Augmenter... in the package Augmenters
which we first neeed to initialize.
It allows us to specify the augmentations to use and in which order to apply them (cf doc)

>>> augmenter = Augmenter1(patch_size_before_final_resize=1000, patch_size_final_resize=256,
...                        allowed_transformations="combinedRotResizeMir_10_0.25_4",
...                        label_access_function=PointAnnotations().get)

Then we can use this object to apply random transformations
to input ⚠️ images (not batch of images).

All of the transformations are randomly chosen and can change between epochs


We made two types of augmenters :

- First type: augment step by step:

>>> image = ...
>>> annotation = ...
>>> image,annotation = augmenter.transform(image,annotation)

- Second type: all augmentations are done at once internally thanks to warpAffine function. It is an optimized version of the
step by step augmentations thanks to warpAffine and potential filters that can be applied on the annotations_patch
before opening and creating the image patch

Typical usage
>>> def get_label_transformed(image_name,transformation_matrix):
...     # get points of polygons for image name
...     # apply the transformation matrix
...     # draw the Annotations map
...     # return the Annotations
>>> partial_transformation_matrix = augmenter.choose_new_augmentations()
>>> for patch_upper_left_corner_coords in augmenter.get_grid(image.shape, partial_transformation_matrix)):
...     annotations_patch,transformation_matrix = augmenter.transform_label(get_label_transformed,"017635_0212D5_0F87",
...                                                                         partial_transformation_matrix,
...                                                                         patch_upper_left_corner_coords)
>>>     image_patch, transformation_matrix = augmenter.transform_image(image,
...                                                                    partial_transformation_matrix,
...                                                                    patch_upper_left_corner_coords
...                                                                   )
"""
