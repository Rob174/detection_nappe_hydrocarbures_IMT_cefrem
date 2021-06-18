import json
from functools import lru_cache
from typing import Tuple

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
import main.src.data.resizer as resizer
import copy

from main.src.data.TwoWayDict import TwoWayDict
from main.src.param_savers.BaseClass import BaseClass


class DataSentinel1Segmentation(BaseClass):
    def __init__(self,limit_num_images=None,input_size=None):
        self.images = File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r")
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json") as fp:
            self.images_infos = copy.deepcopy(json.load(fp))
        self.annotations_labels = File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5", "r")
        self.attr_limit_num_images = limit_num_images
        self.attr_class_mapping = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                1: "spill",
                2: "seep",
            })
        self.attr_resizer = resizer.Resizer(out_size_w=input_size)

        self.attrend_resolutionX_stats = {}
        self.attrend_resolutionY_stats = {}
        self.img_not_seen = self.get_all_items()

    def __getitem__(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        item = self.get_all_items()[id]
        img = self.images[item]
        if item in self.img_not_seen:
            self.save_resolution(item,img)
        self.current_name = item
        return self.attr_resizer(img), self.attr_resizer(self.annotations_labels[item])

    def save_resolution(self,item:str,img:np.ndarray):
        resolution = [v for v in self.images_infos[item]["resolution"]]
        scale_factor = self.attr_resizer.attr_out_size_w / img.shape[1]
        resolution[0] *= scale_factor
        resolution[1] *= scale_factor
        if resolution[0] not in self.attrend_resolutionX_stats.keys():
            self.attrend_resolutionX_stats[resolution[0]] = 0
        self.attrend_resolutionX_stats[resolution[0]] += 1
        if resolution[1] not in self.attrend_resolutionY_stats.keys():
            self.attrend_resolutionY_stats[resolution[1]] = 0
        self.attrend_resolutionY_stats[resolution[1]] += 1
    @lru_cache(maxsize=1)
    def get_all_items(self):
        if self.attr_limit_num_images is not None:
            return list(self.images.keys())[:self.attr_limit_num_images]
        return list(self.images.keys())

    def close(self):
        self.images.close()
        self.annotations_labels.close()

    def __len__(self) -> int:
        return len(self.get_all_items())

# Usage: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html https://pytorch.org/docs/stable/data.html
# data = DataLoader(DataSentinel1Segmentation(), batch_size=1, shuffle=False, sampler=None,
#                   batch_sampler=None, num_workers=0, collate_fn=None,
#                   pin_memory=False, drop_last=False, timeout=0,
#                   worker_init_fn=None, prefetch_factor=2,
#                   persistent_workers=False)

# if __name__ == "__main__":
