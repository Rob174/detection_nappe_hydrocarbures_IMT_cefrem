"""Module containing grid makers. They can create a grid (list of coordinates of the upper left corner of a patch)
and create the transformation matrix to apply to the source image to put the patch at the upper left corner position of the view
One can then slice the provided dimensions of the patch to retrieve it.

It has been thought this way to optimize the usage of the opencv warpAffine function for augmentations

Individual usage: example to create a grid of 256px patches

>>> from main.src.data.GridMaker.GridMaker import GridMaker
>>> import numpy as np
>>> import cv2
>>> grid  = GridMaker(patch_size_final_resize=256)
>>> grid_coords = grid.get_grid(img_shape=(10000,15000),partial_transformation_matrix=np.identity(3))
>>> chosen_patch = grid_coords[10]
>>> grid.get_patch_transformation_matrix(chosen_patch)
>>> patch = cv2.warpAffine(source_image,M=np.identity(3),dsize=256,flags=cv2.INTER_LANCZOS4)

"""
