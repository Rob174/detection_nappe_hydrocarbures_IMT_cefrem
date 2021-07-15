import json
from typing import Tuple

import torch

from main.FolderInfos import FolderInfos
from main.src.data.Augmentation.Augmenters.enums import EnumAugmenter
from main.src.data.balance_classes.enums import EnumBalance
from main.src.data.classification.ClassificationCache import ClassificationCache
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.classification.enums import EnumLabelModifier, EnumClassPatchAdder
from main.src.data.enums import EnumUsage, EnumClasses
from main.src.data.patch_creator.enums import EnumPatchAlgorithm, EnumPatchExcludePolicy
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.param_savers.BaseClass import BaseClass


class DatasetFactory(BaseClass, torch.utils.data.IterableDataset):
    """Class managing the attr_dataset creation and access with options of:
    - different attr_dataset possible
    - different patch creator possible

    Args:
        dataset_name: EnumLabelModifier,
        usage_type: EnumUsage,
        patch_creator: EnumPatchAlgorithm, for classification only
        grid_size: int, classification only with fixed_px size. To specify the size of a patch
        input_size: int, size of the image given to the attr_model
        exclusion_policy: EnumPatchExcludePolicy, policy to exclude patches. See [NoLabelModifier](./classification/NoLabelModifier.html)
        exclusion_policy_threshold: int, parameter for EnumPatchExcludePolicy.MarginMoreThan
        classes_to_use: Tuple[EnumClasses], the classes to use
        balance: EnumBalance,
        margin: int, additionnal parameter to balance classes, cf doc in NoLabelModifier or in BalanceClasses1
        augmentations_img: opt str, list of augmentations to apply seprated by commas
        augmenter_img: opt EnumAugmenter,
        augmentations_patch: opt str, list of augmentations to apply seprated by commas
        augmenter_patch: opt EnumAugmenter,
        augmentation_factor: int, the number of times that the source image is augmented
        force_classifpatch: bool, force to use the class classificattionpatch
        other_class_adder: EnumClassPatchAdder to select classadder object
        interval: int, interval between two un annotated patches
    """

    def __init__(self, dataset_name: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 usage_type: EnumUsage = EnumUsage.Classification,
                 patch_creator: EnumPatchAlgorithm = EnumPatchAlgorithm.FixedPx,
                 grid_size=1000, input_size=1000,
                 exclusion_policy=EnumPatchExcludePolicy.MarginMoreThan, exclusion_policy_threshold: int = 1000,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill),
                 balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentations_patch="none", augmenter_patch: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor=1, force_classifpatch=False,
                 other_class_adder: EnumClassPatchAdder = EnumClassPatchAdder.NoClassPatchAdder,
                 interval: int = 1):
        self.attr_global_name = "data"
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            dico_infos = json.load(fp)

        if usage_type == EnumUsage.Classification:
            # if input_size == 256 and balance == EnumBalance.BalanceClasses1 and augmenter_img == EnumAugmenter.Augmenter1 \
            #         and augmentations_img == "combinedRotResizeMir_10_0.25_4" and augmenter_patch == EnumAugmenter.NoAugmenter \
            #         and augmentations_patch == "none" and exclusion_policy == EnumPatchExcludePolicy.MarginMoreThan and exclusion_policy_threshold == 1000 \
            #         and grid_size == 1000 and not force_classifpatch:
            #     self.attr_dataset = ClassificationCache(label_modifier=dataset_name, classes_to_use=classes_to_use,
            #                                             other_class_adder=other_class_adder,interval=interval)
            # else:
            if patch_creator == EnumPatchAlgorithm.FixedPx:
                self.attr_patch_creator = Patch_creator0(grid_size_px=grid_size,
                                                         images_informations_preprocessed=dico_infos,
                                                         exclusion_policy=exclusion_policy,
                                                         exclusion_policy_threshold=exclusion_policy_threshold)
            else:
                raise NotImplementedError(f"{patch_creator} is not implemented")
            self.attr_dataset = ClassificationPatch(self.attr_patch_creator, input_size=input_size,
                                                    classes_to_use=classes_to_use,
                                                    balance=balance,
                                                    augmentations_img=augmentations_img,
                                                    augmenter_img=augmenter_img,
                                                    augmentations_patch=augmentations_patch,
                                                    augmenter_patch=augmenter_patch,
                                                    augmentation_factor=augmentation_factor,
                                                    label_modifier=dataset_name)


        elif usage_type == EnumUsage.Segmentation:
            if dataset_name == "sentinel1":
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __iter__(self, dataset):
        return self.attr_dataset.__iter__(dataset)

    def len(self, dataset):
        return self.attr_dataset.len(dataset)


if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name=EnumLabelModifier.NoLabelModifier,
                                     usage_type=EnumUsage.Classification, patch_creator=EnumPatchAlgorithm.FixedPx,
                                     grid_size=1000, input_size=256)

    for input, output, transformation_matrix, name in dataset_factory:
        print("done")
