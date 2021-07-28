import json
from enum import Enum

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.enums import EnumAugmenter, EnumBalance, EnumLabelModifier, EnumUsage, EnumClasses, \
    EnumPatchExcludePolicy, EnumPatchAlgorithm

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


    class Cases(str, Enum):
        SeepOnly = "seep_only"
        SpillOnly = "spill_only"
        SeepSpill = "seep_spill"


    class UnwantedCaseException(Exception):
        def __init__(self, output):
            super(UnwantedCaseException, self).__init__(
                f"Case with seep to {output[0]} and spill to {output[1]} is not supported")


    class WrapperIter:
        def __init__(self, dataset_factory, dataset):
            self.dataset_factory = dataset_factory
            self.dataset = dataset

        def __iter__(self):
            return self.dataset_factory.__iter__(self.dataset)


    dico_eff = {Cases.SeepOnly: 0, Cases.SpillOnly: 0, Cases.SeepSpill: 0}
    for dataset in ["tr", "valid"]:
        wrapper = WrapperIter(dataset_factory, dataset)
        for [input, output, transformation_matrix, name] in wrapper:
            if output[0] == 1 and output[1] == 1:
                dico_eff[Cases.SeepSpill] += 1
            elif output[0] == 1:
                dico_eff[Cases.SeepOnly] += 1
            elif output[1] == 1:
                dico_eff[Cases.SpillOnly] += 1
            else:
                raise UnwantedCaseException(output)
    with open(FolderInfos.input_data_folder + "filtered_cache_classes.json", "w") as fp:
        json.dump(dico_eff, fp)
