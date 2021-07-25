"""
# Quickstart

## Automatic configuration

In this project the user can directly use the main_script file to train a desired model on a dataset.
To choose all of the options the user can provide parameters in the console or can create a custom Parser
with default value according to  his/her choice. An premade parser is provided in  the main/src/parsers module.

We can also want to modify some options for the system. That is why we can have the following questions:

**I.**      How to supply my own dataset ?

**II.**     How create patches outside of the system ?

**III.**    How to create a custom training process ?

**IV.**     How to predict on new patches ?

## I. How to supply my own dataset

The system requires 3 subdatasets to build a dataset: the image subdataset, the corresponding annotations (2d annotations)
and an information file.


### Building the files

### Images

They must be provided as np.ndarrays in an hdf5 file following the same method as for annotations


#### Annotations

Annotations can be provided as np.ndarrays or polygons respectively to store in an hdf5 or pickle file

##### HDF5

The user has to create one hdf5 dataset (with the create_dataset function, cf hdf5 package doc) per image.
The name of the dataset must be an uniq id

##### Polygons

Stored under a dict form:

>>> {
...     "id_image_1": [
...         polygon1,
...         polygon2,
...         ...
...     ]
... }

with polygon a dict with:

>>> polygon1 = {
...     "label":"...", #(ex: seep, spill...),
...     "points":[(...,...),(...,...),...]
... }

#### Information file structure

The information file is a json file containing at least:
- for a dataset to build patch (as for images_informations_preprocessed.json):
  - under the key "transformation" the transformation matrix from geographic coordinates to pixel coordinates as proposed by rasterio

>>> {
...     "image_id1":{
...         "transformation":...,
...         ...
...     },
...     ...
... }

- for a dataset of premade patches (as for filtered_img_infos.json):
  - under the key "transformation_matrix" the transformation matrix from the source image to the patch (with augmentations)
  - under the key "source_img": the id in the origin dataset from which the patch has been built

>>> {
...     "image_id1":{
...         "transformation_matrix":...,
...         "source_img":"..."
...     },
...     ...
... }

### Managing the files in the system

#### 1. Build subdatasets

It is mandatory to create/use a class to manage the image and annotation subdataset.

The user can use premade classes  of the [main/src/data/Datasets module](./data/Datasets/index.html) to manage hdf5 files or polygon annotations

The user can also build a custom dataset y following the structure of the [AbstractDataset class](./data/Datasets/AbstractDataset.html)

⚠⚠ go to the original python code to build your own as this documentation does not allow to show python magic functions. Some of them are required to implement for a custom dataset


#### 2. Build a fabric

To associate the subdatasets together and open the json information file the user has to create a Fabric class creating all of them.

The expected structure is following the [AbstractFabricDatasets class](./data/Datasets/Fabrics/AbstractFabricDatasets.html)

### Providing it to the system


The user has first to provide the relevant arguments to the [DatasetFactory constructor](./data/DatasetFactory.html) for the [ClassificationGeneratorCache](./data/classification/ClassificationGeneratorCache.html)
or [ClassificationGeneratorPatch](|./data/classification/ClassificationGeneratorPatch.html) class

The user can specify in this same constructor the algorithm to use with the `choose_dataset` argument and pass it the result
of the custom made Factory call function (the user can use arguments unpacking to use less lines of code)

>>> dataset_factory.set_datasets(*CustomDatasetFactory()())


## II. How to create patches outside of the system

The user can create patches on preopened image by using the [NoAugmenter](./data/Augmentation/Augmenters/NoAugmenter.py) class

>>> from main.src.data.Augmentation.Augmenters.NoAugmenter import NoAugmenter
>>> augmenter = NoAugmenter(patch_size_before_final_resize=1000,patch_size_final_resize=256)

We can then get the partial transformation matrix for the size reduction:

>>> partial_transformation_matrix = augmenter.choose_new_augmentations(custom_image)


Then we have to choose the coordinates of the patch to extract.
For that we can get a grid (list of upper left corners) of patch by calling

>>> grid = augmenter.get_grid(custom_image.shape,partial_transformation_matrix)

Then we can use the augmenter to apply the transformation to get the patch desired in the upper left corner of the view.

>>> augmenter.transform_image(custom_image,partial_transformation_matrix,patch_upper_left_corner_coords)[:256,:256]

## III. How to create a custom training process

The user has a premade training object [Trainer0](./training/Trainers/Trainer0.html).
We can provide it the DatasetFactory with a custom dataset, a custom model, a custom optimizer ....
more details in [Trainer0](./training/Trainers/Trainer0.html) constructor

The user can also build a custom trainer and using the object relevant to the goal pursued.

No specific structure is required. The user can iterator with a for loop over the dataset factory object (see Trainer0 example)
and has to keep in mind that the values produced by the Datasetfactory for loop correspond to individual samples, not batches.

It is the responsability of the Trainer object to build batches for the moment. You can use the Trainer0 structure as an example

## IV. How to build and predict on new patches ?

The user can build new patches by using the datasetfacotry and calling

>>> dataset_factory.attr_dataset.get_patch(....)
patch_img,patch_annotation,transformation_matrix

The arguments then differs for the type of dataset create (ClassificationGeneratorCache or ClassificationGeneratorPatch)

Refers to their documentation to get specific arguments required

Then one can get the model of the ModelFactory by using its model property

>>> my_pytorch_model = model_factory.model

and can then predict the result as usual in pytorch
>>> import torch
>>> with torch.no_grad():
...     my_pytorch_model.eval()
...     input_gpu = torch.Tensor(patch_img).to(torch.device("cuda"))
...     prediction = my_pytorch_model(input_gpu).cpu().detach().numpy()
"""
