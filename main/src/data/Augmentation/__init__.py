

"""
The goal of this module is to manage and apply augmentations.

As we will apply the same types of augmentations to all of the images,
we have a first object of the package Augmenters
which we first needs to be initialized.

Example with Augmenter1

>>> from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
>>> augmenter = Augmenter1(patch_size_before_final_resize=1000, patch_size_final_resize=256,
...                        allowed_transformations=["combinedRotResizeMir_10_0.25_4"])

It allows us to specify the augmentations to use(cf doc) for a source image provided as np.ndarray and
returns the transformation matrix combination of the affine transformations chosen

>>> partial_transformation_matrix = augmenter.choose_new_augmentations(image)

Then we have to choose the coordinates of the patch to extract.
For that we can get a grid (list of upper left corners) of patch by calling

>>> grid = augmenter.get_grid(image.shape,partial_transformation_matrix)

Then we can use the augmenter to apply random transformations to input ️ samples (⚠⚠ not in a batch ⚠⚠).
The input can have different types as an array or a list of points for instance (for labels)

>>> augmenter.transform_image(image,partial_transformation_matrix,patch_upper_left_corner_coords)

We can do the same for the label with the `transform_image` method

All of the transformations are randomly chosen and can change between epochs

"""
