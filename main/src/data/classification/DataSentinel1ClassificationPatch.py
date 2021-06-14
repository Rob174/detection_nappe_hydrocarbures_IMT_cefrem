from functools import lru_cache
from typing import Tuple, List
import numpy as np
import psutil

from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation


class DataSentinel1ClassificationPatch(DataSentinel1Segmentation):
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None):
        self.patch_creator = patch_creator
        self.attr_limit_num_images = limit_num_images
        self.attr_resizer = Resizer(out_size_w=input_size)
        super(DataSentinel1ClassificationPatch, self).__init__(limit_num_images, input_size=input_size)

    @lru_cache(maxsize=1)
    def get_all_items(self):
        list_items = []
        for img_name in list(self.images.keys()):
            img = self.images[img_name]
            num_ids = self.patch_creator.num_available_patches(img)
            list_items.extend([(img_name, i) for i in range(num_ids)])
        if self.attr_limit_num_images is not None:
            return list_items[:self.attr_limit_num_images]
        return list_items

    def __getitem__(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        [item, patch_id] = self.get_all_items()[id]
        img = self.images[item]
        annotations = self.annotations_labels[item]
        img_patch = self.patch_creator(img, item, patch_id=patch_id)
        annotations_patch = self.patch_creator(annotations, item, patch_id=patch_id)

        if (item, patch_id) in self.img_not_seen:
            self.save_resolution(item, img_patch)
        input = self.attr_resizer(img_patch)
        input = np.stack((input, input, input), axis=0)
        return input, self.make_classification_label(annotations_patch)

    def make_classification_label(self, annotations_patch):
        values = np.unique(annotations_patch)
        output = np.zeros((len(self.class_mappings)))
        for i in values:
            output[int(i)] = 1.
        return output

    def make_patches_of_image(self, name: str):
        last_image = np.array(self.images[name], dtype=np.float32)
        liste_patches = []
        num_patches = self.patch_creator.num_available_patches(last_image)
        for id in range(num_patches):
            print(psutil.virtual_memory())
            liste_patches.append([self.patch_creator(last_image, name, patch_id=id, keep=True)])
        annotations = np.array(self.annotations_labels[name], dtype=np.float32)
        for id in range(num_patches):
            patch = self.patch_creator(annotations, name, patch_id=id)
            liste_patches[id].append(self.make_classification_label(patch))
        return liste_patches

    def __len__(self) -> int:
        return len(self.get_all_items())

# Tests in DatasetFactory
