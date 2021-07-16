import json
import numpy as np
from enum import Enum

from h5py import File

from main.FolderInfos import FolderInfos

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    class Cases(str,Enum):
        Mean = "mean"
        Std = "std"
        Sum = "sum_all_vals"
        Eff = "total_num_px"
        DiffMean = "sum_squared_diff_mean"

    dico_stats = {k:0 for k in list(Cases)}
    with File(FolderInfos.input_data_folder+"filtered_other_cache_images.hdf5","r") as cache:
        for input in cache.values():
            dico_stats[Cases.Sum] += np.sum(input)
            dico_stats[Cases.Eff] += input.shape[0]*input.shape[1]
    dico_stats[Cases.Mean] = dico_stats[Cases.Sum] / dico_stats[Cases.Eff]
    with File(FolderInfos.input_data_folder+"filtered_other_cache_images.hdf5","r") as cache:
        for input in cache.values():
            dico_stats[Cases.DiffMean] += np.sum(np.power(input-dico_stats[Cases.Mean],2))
    dico_stats[Cases.Std] = (dico_stats[Cases.DiffMean] / dico_stats[Cases.Eff])**0.5
    with open(FolderInfos.input_data_folder+"filtered_cache_other_pixels_stats.json","w") as fp:
        json.dump(dico_stats,fp)