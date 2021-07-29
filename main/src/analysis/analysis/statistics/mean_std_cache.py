import json
from enum import Enum

import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache
from main.src.data.Datasets.Fabrics.FabricFilteredCacheOther import FabricFilteredCacheOther
from main.src.enums import EnumAugmenter, EnumBalance, EnumLabelModifier, EnumUsage, EnumClasses, \
    EnumPatchExcludePolicy, EnumPatchAlgorithm

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    dataset_name = "filtered_cache_other"
    images,annotations,infos = FabricFilteredCacheOther()()


    class Cases(str, Enum):
        Mean = "mean"
        Std = "std"
        Sum = "sum_all_vals"
        Eff = "total_num_px"
        DiffMean = "sum_squared_diff_mean"


    class UnwantedCaseException(Exception):
        def __init__(self, output):
            super(UnwantedCaseException, self).__init__(
                f"Case with seep to {output[0]} and spill to {output[1]} is not supported")


    dico_stats = {k: 0 for k in list(Cases)}
    for name in images.keys():
        input = images.get(name)
        dico_stats[Cases.Sum] += np.sum(input)
        dico_stats[Cases.Eff] += input.shape[0] * input.shape[1]
    dico_stats[Cases.Mean] = dico_stats[Cases.Sum] / dico_stats[Cases.Eff]
    for name in images.keys():
        input = images.get(name)
        dico_stats[Cases.DiffMean] += np.sum(np.power(input - dico_stats[Cases.Mean], 2))
    dico_stats[Cases.Std] = (dico_stats[Cases.DiffMean] / dico_stats[Cases.Eff]) ** 0.5
    with open(FolderInfos.input_data_folder + dataset_name+"_pixels_stats.json", "w") as fp:
        json.dump(dico_stats, fp)
