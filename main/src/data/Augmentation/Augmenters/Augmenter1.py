from main.src.data.Augmentation.Augmentations.Mirrors import Mirrors
from main.src.data.Augmentation.Augmentations.Resize import Resize
from main.src.data.Augmentation.Augmentations.Rotations import Rotations
from main.src.data.Augmentation.Augmenters.Augmenter0 import Augmenter0
from main.src.param_savers.BaseClass import BaseClass

import numpy as np
from typing import Tuple

class Augmenter1(Augmenter0):
    """Manage and keep track of augmentations to apply on source images only to directly extract patches

        Args:
            allowed_transformations: str, list of augmentations to apply seprated by commas. Currently supported:
            - mirrors
            - rotations
            - resize: provided as "resize_...range..._...shift..."
    """
    def __init__(self,allowed_transformations="mirrors,rotations"):
        super(Augmenter1, self).__init__(allowed_transformations)
    def add_transformation(self,allowed_transformations):
        """Method that map transformation names with actual classes. Splited from the __init__ to be able to overload

        Args:
            allowed_transformations: str, list of augmentations to apply seprated by commas. Currently supported:
            - rotation_resize_{rotation_step}_{resize_lower_fact_float}_{resize_upper_fact_float} (has to be selected to extract the patch
            - mirrors (has to be specified after rotation_resize to consume less memory)

        Returns:

        """
        seen = False
        for transfo in self.attr_allowed_transformations.split(","):
            if transfo == "mirrors":
                self.attr_transformations_classes.append(Mirrors())
            elif "rotation_resize" in transfo:
                seen = True
                [_,_,rotation_step,resize_lower_fact_float,resize_upper_fact_float] = transfo.split("_")
                rotation_step = float(rotation_step)
                resize_lower_fact_float = float(resize_lower_fact_float)
                resize_upper_fact_float = float(resize_upper_fact_float)
                self.attr_transformations_classes.append(Resize(range=range,shift=shift))
            else:
                raise NotImplementedError(f"{transfo} is not implemented")
        if seen is False:
            raise Exception("rotation_resize augmentation has to be always selected to compute patches with this augmenter. Use Augmenter0 otherwise")
    def transform(self,image: np.ndarray, annotation: np.ndarray,
                  patch_upper_left_corner_shift: Tuple[int],  ) -> Tuple[np.ndarray,np.ndarray]: