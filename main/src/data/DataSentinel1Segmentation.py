from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np
from  h5py import File

from main.FolderInfos import FolderInfos


class DataGet:
    def __init__(self):
        self.images = File(FolderInfos.base_folder,"r")
    def __getitem__(self, item) -> Tuple[np.ndarray, np.ndarray]:

    def __len__(self) -> int:

data = DataLoader(DataGet(), batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, prefetch_factor=2,
           persistent_workers=False)