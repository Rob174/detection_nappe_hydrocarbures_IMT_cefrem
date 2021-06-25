from main.src.data.Augmentation.Augmentations.Mirrors import Mirrors
from main.src.data.Augmentation.Augmentations.Resize import Resize
from main.src.data.Augmentation.Augmentations.Rotations import Rotations
from main.src.param_savers.BaseClass import BaseClass

import numpy as np
from typing import Tuple

class Augmenter0(BaseClass):
    """Manage and keep track of augmentations to apply

        Args:
            allowed_transformations: str, list of augmentations to apply seprated by commas. Currently supported:
            - mirrors
            - rotations
            - resize: provided as "resize_...range..._...shift..."
    """
    def __init__(self,allowed_transformations="mirrors,rotations"):
        self.attr_allowed_transformations = allowed_transformations
        self.attr_transformations_classes =  []
        for transfo in self.attr_allowed_transformations.split(","):
            if transfo == "mirrors":
                self.attr_transformations_classes.append(Mirrors.compute_random_augment)
            elif transfo == "rotations":
                self.attr_transformations_classes.append(Rotations.compute_random_augment)
            elif "resize" in transfo:
                [_,range,shift] = transfo.split("_")
                range = float(range)
                shift = float(shift)
                self.attr_transformations_classes.append(lambda x,y:Resize.compute_random_augment(x,y,range=range,shift=shift))
            else:
                raise NotImplementedError(f"{transfo} is not implemented")
    def transform(self,image: np.ndarray, annotation: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """Compute the random augmentations in the order in which they have been supplied.

        Apply the same augmentations to the image and to the annotation

        Args:
            image: np.ndarray, the input image to transform
            annotation: np.array, the corresponding annotation

        Returns:
            The randomly transformed image and annotation

        """
        for compute_random_augment in self.attr_transformations_classes:
            image,annotation = compute_random_augment(image,annotation)
        return image,annotation