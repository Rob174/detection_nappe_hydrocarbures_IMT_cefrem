import json
from typing import Tuple

import numpy as np
from h5py import File
from torch.utils.data import DataLoader

from main.FolderInfos import FolderInfos


class DataGet:
    def __init__(self):
        self.images = File(f"{FolderInfos.input_data_folder}images.hdf5", "r")
        with open(f"{FolderInfos.input_data_folder}images_infos.json") as fp:
            self.images_infos = json.load(fp)
        self.annotations_labels = File(f"{FolderInfos.input_data_folder}annotations_labels.hdf5", "r")

    def __getitem__(self, item: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.images[item], self.annotations_labels[item]

    def __len__(self) -> int:
        return min(len(self.images.keys()), len(self.annotations_labels.keys()), len(self.images_infos.keys()))


data = DataLoader(DataGet(), batch_size=1, shuffle=False, sampler=None,
                  batch_sampler=None, num_workers=0, collate_fn=None,
                  pin_memory=False, drop_last=False, timeout=0,
                  worker_init_fn=None, prefetch_factor=2,
                  persistent_workers=False)