"""Classes that allows to generate values from an other dataset than the main one.
The philosophy is the following:

We have a main group of datasets that we can specify as in the Fabrics (FabricFilteredCache, ...).
Then, We  want to introduce another dataset during the training process. For that we can use the PatchAdder classes
For that we have to choose the number of the main dataset samples to wait before introducing a new sample from this new dataset.

Used in the internship project to balance a dataset by splitting it into a main dataset and another PatchAdder dataset
to add patches without any annotation
"""
