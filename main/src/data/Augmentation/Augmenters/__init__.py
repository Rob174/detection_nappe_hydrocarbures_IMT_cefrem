"""
Augmenters classes which manage and apply augmentations.

For good practices purposes, these classes have to store informations about which augmentations are applied,
eventually how these augmentation are applied.... with the attribute attr_ syntax (cf saver class)

They have to provide at least the following interface:

>>> class MyAugmenter(BaseClass):
...     def __init__(self,*args,**kargs):
...         # store informations about transformations in attr_... attributes
...     def transform(self,image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
...         # ...Apply the same transformation to the image and annotation...
...         return image,annotation
"""