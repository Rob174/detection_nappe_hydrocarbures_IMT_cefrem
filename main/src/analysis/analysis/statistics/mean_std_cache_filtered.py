import json
import numpy as np
from enum import Enum

from main.FolderInfos import FolderInfos
from main.src.enums import EnumAugmenter
from main.src.data.DatasetFactory import DatasetFactory
from main.src.enums import EnumBalance
from main.src.enums import EnumLabelModifier
from main.src.enums import EnumUsage, EnumClasses
from main.src.enums import EnumPatchExcludePolicy, EnumPatchAlgorithm

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    dataset_factory = DatasetFactory(dataset_name=EnumLabelModifier.LabelModifier1,
                             usage_type=EnumUsage.Classification,
                             patch_creator=EnumPatchAlgorithm.FixedPx,
                             grid_size=1000,
                             input_size=256,
                             exclusion_policy=EnumPatchExcludePolicy.MarginMoreThan,
                             exclusion_policy_threshold=1000,
                             classes_to_use=(EnumClasses.Seep, EnumClasses.Spill),
                             balance=EnumBalance.BalanceClasses1,
                             augmenter_img=EnumAugmenter.Augmenter1,
                             augmentations_img="combinedRotResizeMir_10_0.25_4",
                             augmenter_patch=EnumAugmenter.NoAugmenter,
                             augmentations_patch="none",
                             augmentation_factor=100
                             )
    class Cases(str,Enum):
        Mean = "mean"
        Std = "std"
        Sum = "sum_all_vals"
        Eff = "total_num_px"
        DiffMean = "sum_squared_diff_mean"
    class UnwantedCaseException(Exception):
        def __init__(self,output):
            super(UnwantedCaseException, self).__init__(f"Case with seep to {output[0]} and spill to {output[1]} is not supported")
    class WrapperIter:
        def __init__(self,dataset_factory,dataset):
            self.dataset_factory = dataset_factory
            self.dataset = dataset
        def __iter__(self):
            return self.dataset_factory.__iter__(self.dataset)
    dico_stats = {k:0 for k in list(Cases)}
    for dataset in ["tr","valid"]:
        wrapper = WrapperIter(dataset_factory,dataset)
        for [input,output,transformation_matrix,name] in wrapper:
            dico_stats[Cases.Sum] += np.sum(input)
            dico_stats[Cases.Eff] += input.shape[0]*input.shape[1]
    dico_stats[Cases.Mean] = dico_stats[Cases.Sum] / dico_stats[Cases.Eff]
    for dataset in ["tr","valid"]:
        wrapper = WrapperIter(dataset_factory,dataset)
        for [input,output,transformation_matrix,name] in wrapper:
            dico_stats[Cases.DiffMean] += np.sum(np.power(input-dico_stats[Cases.Mean],2))
    dico_stats[Cases.Std] = (dico_stats[Cases.DiffMean] / dico_stats[Cases.Eff])**0.5
    with open(FolderInfos.input_data_folder+"filtered_cache_pixels_stats.json","w") as fp:
        json.dump(dico_stats,fp)