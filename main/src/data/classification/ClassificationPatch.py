from functools import lru_cache
from typing import Tuple, List, Union
import numpy as np
import psutil

from main.src.data.TwoWayDict import  Way
from main.src.data.balance_classes.balance_classes import BalanceClasses1
from main.src.data.balance_classes.no_balance import NoBalance
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import time


class ClassificationPatch(DataSentinel1Segmentation):
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None, balance="nobalance",margin=None):
        self.attr_name = self.__class__.__name__
        self.patch_creator = patch_creator
        self.attr_limit_num_images = limit_num_images
        self.attr_resizer = Resizer(out_size_w=input_size)
        super(ClassificationPatch, self).__init__(limit_num_images, input_size=input_size)
        self.attr_global_name = "dataset"
        self.reject = False
        if balance == "nobalance":
            self.attr_balance = NoBalance()
        elif balance == "balanceclasses1":
            self.attr_balance = BalanceClasses1(classes_indexes=self.attr_class_mapping.keys(Way.ORIGINAL_WAY),
                                                margin=margin)


    @lru_cache(maxsize=1)
    def get_all_items(self):
        list_items = []
        for img_name in list(self.images.keys()):
            img = self.images[img_name]
            num_ids = self.patch_creator.num_available_patches(img)
            list_items.extend([[img_name, i] for i in range(num_ids)])
        if self.attr_limit_num_images is not None:
            return list_items[:self.attr_limit_num_images]
        return list_items

    def __getitem__(self, id: Union[int,List[int]]) -> Tuple[np.ndarray, np.ndarray,bool]: # btwn 25 and 50 ms
        [item, patch_id] = self.get_all_items()[id] # 0 ns
        img = self.images[item] # 1ms but 0 most of the time
        annotations = self.annotations_labels[item] # 1ms but 0 most of the time
         # two lines: btwn 21 and 54 ms
        img_patch,reject = self.patch_creator(img, item, patch_id=patch_id) # btwn 10 ms and 50 ms
        annotations_patch,reject = self.patch_creator(annotations, item, patch_id=patch_id) # btwn 10 ms and 30 ms (10 ms most of the time)
        if (item, patch_id) in self.img_not_seen: # Gpe of 2 lines: ~ 1 ms
            self.save_resolution(item, img_patch) #
        input = self.attr_resizer(img_patch) # ~ 0 ns most of the time, 1 ms sometimes
        input = np.stack((input, input, input), axis=0) # 0 ns most of the time
        classif,balance_reject = self.make_classification_label(annotations_patch) # ~ 2 ms
        reject = reject and balance_reject
        return input, classif, reject

    def make_classification_label(self, annotations_patch):
        output = np.zeros((len(self.attr_original_class_mapping),),dtype=np.float32) # 0 ns
        for value in self.attr_class_mapping.keys(Way.ORIGINAL_WAY): # btwn 1 and 2 ms
            value = int(value)
            if value in annotations_patch:
                output[value] = 1.
        balance_reject = self.attr_balance.filter(output)
        return output,balance_reject

    def make_patches_of_image(self, name: str):
        last_image = np.copy(np.array(self.images[name], dtype=np.float32))
        liste_patches = []
        num_patches = self.patch_creator.num_available_patches(last_image)
        for id in range(num_patches):
            patch,reject = self.patch_creator(last_image, name, patch_id=id, keep=True)
            liste_patches.append([patch])
        annotations = np.array(self.annotations_labels[name], dtype=np.float32)
        for id in range(num_patches):
            patch,reject = self.patch_creator(annotations, name, patch_id=id)
            liste_patches[id].append(self.make_classification_label(patch))
            liste_patches[id].append(reject)
        return liste_patches

    def __len__(self) -> int:
        return len(self.get_all_items())

# Tests in DatasetFactory
