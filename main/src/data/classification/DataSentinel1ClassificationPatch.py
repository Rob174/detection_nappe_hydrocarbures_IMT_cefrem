from typing import Tuple
import numpy as np

from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation


class DataSentinel1ClassificationPatch(DataSentinel1Segmentation):
    def __init__(self,class_mappings_file: str, path_creator):
        super()
        self.patch_creator = path_creator

    def __getitem__(self, item: str, patch_id: int) -> Tuple[np.ndarray, np.ndarray]:
        img = self.images[item]
        num_patches = self.patch_creator(img,patch_id)
        annotations = self.annotations_labels[item]
        img_patches = self.patch_creator(img,item)
        annotations_patches = self.patch_creator(annotations, item)
        return img_patches,annotations_patches