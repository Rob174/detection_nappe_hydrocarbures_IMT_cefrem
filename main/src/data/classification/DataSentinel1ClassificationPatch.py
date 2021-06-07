from functools import lru_cache
from typing import Tuple
import numpy as np

from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation


class DataSentinel1ClassificationPatch(DataSentinel1Segmentation):
    def __init__(self, patch_creator: Patch_creator0,limit_num_images=None):
        super(DataSentinel1ClassificationPatch, self).__init__(limit_num_images)
        self.patch_creator = patch_creator
        self.limit_num_images = limit_num_images

    @lru_cache(maxsize=1)
    def get_all_items(self):
        list_items = []
        for img_name in list(self.images.keys()):
            img = self.images[img_name]
            num_ids = self.patch_creator.num_available_patches(img)
            list_items.extend([(img_name,i) for i in range(num_ids)])
        if self.limit_num_images is not None:
            return list_items[:self.limit_num_images]
        return list_items

    def __getitem__(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        [item,patch_id] = self.get_all_items()[id]
        img = self.images[item]
        annotations = self.annotations_labels[item]
        img_patches = self.patch_creator(img,item,patch_id=patch_id)
        annotations_patches = self.patch_creator(annotations, item,patch_id=patch_id)
        return img_patches,annotations_patches

    def __len__(self) -> int:
        return len(self.get_all_items())

# Tests in DatasetFactory