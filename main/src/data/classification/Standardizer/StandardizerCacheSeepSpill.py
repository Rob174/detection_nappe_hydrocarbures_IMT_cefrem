import json
import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.classification.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.param_savers.BaseClass import BaseClass


class StandardizerCacheSeepSpill(BaseClass,AbstractStandardizer):
    def __init__(self):
        with open(FolderInfos.input_data_folder+"filtered_cache_pixels_stats.json","r") as fp:
            self.stat_dico = json.load(fp)
    @property
    def mean(self):
        return self.stat_dico["mean"]
    @property
    def std(self):
        return self.stat_dico["std"]
    @property
    def n(self) -> int:
        return self.stat_dico["total_num_px"]
    def standardize(self,image: np.ndarray):
        return (image-self.mean)/self.std