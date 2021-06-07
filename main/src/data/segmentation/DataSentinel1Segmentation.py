import json
from functools import lru_cache
from typing import Tuple

import numpy as np
from h5py import File
from torch.utils.data import DataLoader

from main.FolderInfos import FolderInfos


class DataSentinel1Segmentation:
    def __init__(self,limit_num_images=None):
        self.images = File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r")
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json") as fp:
            self.images_infos = json.load(fp)
        self.annotations_labels = File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5", "r")
        self.limit_num_images = limit_num_images

    def __getitem__(self, id: int) -> Tuple[np.ndarray, np.ndarray]:
        item = self.get_all_items()[id]
        return self.images[item], self.annotations_labels[item]


    @lru_cache(maxsize=1)
    def get_all_items(self):
        if self.limit_num_images is not None:
            return list(self.images.keys())[:self.limit_num_images]
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
