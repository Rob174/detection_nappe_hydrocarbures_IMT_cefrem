import json, random

import numpy as np

from h5py import File
from typing import Optional, Tuple

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass


class OtherClassPatchAdder(BaseClass):
    """Generate determined number of patches with only the other class

    Args:
        interval: int, interval between two un annotated patches
    """
    def __init__(self,interval):
        self.attr_interval = interval
        self.cache_img = File(FolderInfos.input_data_folder+"filtered_other_cache_images.hdf5","r")
        with open(FolderInfos.input_data_folder+"filtered_other_img_infos.json","r") as fp:
            self.cache_infos = json.load(fp)
        self.num_annotated_btwn = 0

        self.ordered_keys = list(self.cache_img.keys())
        self.keys_iterator = None
    def reinitialize_iter(self):
        random.shuffle(self.ordered_keys)
        self.keys_iterator = iter(self.ordered_keys)
    def next_index(self):
        try:
            id = next(self.keys_iterator)
        except (StopIteration, TypeError):
            self.reinitialize_iter()
            id = next(self.keys_iterator)
        return id


    def generate_if_required(self) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray,str]]:
        if self.num_annotated_btwn == self.attr_interval:
            self.num_annotated_btwn = 0
            id = self.next_index()
            patch = np.array(self.cache_img[id],dtype=np.float32)[:,0,:,:]
            annotation = np.zeros(patch.shape,dtype =np.float32)
            transformation_matrix = np.array(self.cache_infos[id]["transformation_matrix"])
            return patch,annotation,transformation_matrix,self.cache_infos[id]["source_img"]
        else:
            self.num_annotated_btwn += 1
            return None



