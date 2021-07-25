"""

Object to build custom datasets.

For image or vectors it is adviced to use hdf5 file. This type of datasets allow you to build a dataset using the [ImageDataset](./ImageDataset.html) object

For annotations stored as polygons it is adviced to build a pickle file and to store inside a list with the folllowing structure:

>>> {
...     "id_img_1": [
...         {  # Represent 1 polygon on the image
...             "points":list_of_points_of_polygon,
...             "label":"my_name"
...         }, ...
...     ],
... }

It allows to use the AugmentationApplierLabelPoints class. Otherwise another augmentationApplier class has to be created

For code clarity and as we combine multiple datasets for the training (image dataset, annotation dataset and image informations)
we advice to create a Fabric as in the main.src.data.Datasets.Fabrics module

This Fabric must follow the [AbstractFabricDatasets](./Fabrics/AbstractFabricDatasets.html) class prototypes. It has to return:

- an image dataset
- an annotation dataset (image or points)
- an image information dict with different requiredment depending of the type of dataset (to be improved in the future.
Migration script will be provided). Use the image information json file of the [FabricFilteredCache](./Fabrics/FabricFilteredCache.html)
and the [FabricPreprocessedCache](./Fabrics/FabricPreprocessedCache.html) class to build your own.
"""