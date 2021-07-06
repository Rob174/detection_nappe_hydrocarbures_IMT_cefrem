from main.FolderInfos import FolderInfos
from main.src.data.Augmentation.Augmenters.enums import EnumAugmenter
from main.src.data.balance_classes.enums import EnumBalance
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.classification.ClassificationPatch1 import ClassificationPatch1
from main.src.data.classification.ClassificationPatch2 import ClassificationPatch2
from main.src.data.classification.enums import EnumClassificationDataset
from main.src.data.enums import EnumUsage
from main.src.data.patch_creator.enums import EnumPatchAlgorithm, EnumPatchExcludePolicy
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import matplotlib.pyplot as plt
import json
import plotly.express as px
import pandas as pd
from main.src.param_savers.BaseClass import BaseClass
import torch
import time


class DatasetFactory(BaseClass, torch.utils.data.IterableDataset):
    """Class managing the dataset creation and access with options of:
    - different dataset possible
    - different patch creator possible

    Args:
        dataset_name: EnumClassificationDataset,
        usage_type: EnumUsage,
        patch_creator: EnumPatchAlgorithm, for classification only
        grid_size: int, classification only with fixed_px size. To specify the size of a patch
        input_size: int, size of the image given to the model
        exclusion_policy: EnumPatchExcludePolicy, policy to exclude patches. See [ClassificationPatch](./classification/ClassificationPatch.html)
        exclusion_policy_threshold: int, parameter for EnumPatchExcludePolicy.MarginMoreThan
        classes_to_use: str, classes names separated but commas to indicate the classes to use
        balance: EnumBalance,
        margin: int, additionnal parameter to balance classes, cf doc in ClassificationPatch or in BalanceClasses1
        augmentations_img: opt str, list of augmentations to apply seprated by commas
        augmenter_img: opt EnumAugmenter,
        augmentations_patch: opt str, list of augmentations to apply seprated by commas
        augmenter_patch: opt EnumAugmenter,
        augmentation_factor: int, the number of times that the source image is augmented
    """

    def __init__(self, dataset_name: EnumClassificationDataset = EnumClassificationDataset.ClassificationPatch,
                 usage_type: EnumUsage = EnumUsage.Classification,
                 patch_creator: EnumPatchAlgorithm = EnumPatchAlgorithm.FixedPx,
                 grid_size=1000, input_size=1000,
                 exclusion_policy=EnumPatchExcludePolicy.MarginMoreThan,exclusion_policy_threshold:int=1000,
                 classes_to_use="seep,spills",
                 balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentations_patch="none", augmenter_patch: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor=1):
        self.attr_global_name = "data"
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            dico_infos = json.load(fp)
        if patch_creator == EnumPatchAlgorithm.FixedPx:
            self.attr_patch_creator = Patch_creator0(grid_size_px=grid_size,
                                                     images_informations_preprocessed=dico_infos,
                                                     exclusion_policy=exclusion_policy,
                                                     exclude_policy_threshold=exclusion_policy_threshold)
        else:
            raise NotImplementedError(f"{patch_creator} is not implemented")

        if usage_type == EnumUsage.Classification:
            if dataset_name == EnumClassificationDataset.ClassificationPatch:
                self.attr_dataset = ClassificationPatch(self.attr_patch_creator, input_size=input_size,
                                                        balance=balance,
                                                        augmentations_img=augmentations_img,
                                                        augmenter_img=augmenter_img,
                                                        augmentations_patch=augmentations_patch,
                                                        augmenter_patch=augmenter_patch,
                                                        augmentation_factor=augmentation_factor)
            elif dataset_name == EnumClassificationDataset.ClassificationPatch1:
                self.attr_dataset = ClassificationPatch1(self.attr_patch_creator, input_size=input_size,
                                                         classes_to_use=classes_to_use,
                                                         balance=balance,
                                                         augmentations_img=augmentations_img,
                                                         augmenter_img=augmenter_img,
                                                         augmentations_patch=augmentations_patch,
                                                         augmenter_patch=augmenter_patch,
                                                         augmentation_factor=augmentation_factor)
            elif dataset_name == EnumClassificationDataset.ClassificationPatch2:
                self.attr_dataset = ClassificationPatch2(self.attr_patch_creator, input_size=input_size,
                                                         classes_to_use=classes_to_use,
                                                         balance=balance,
                                                         augmentations_img=augmentations_img,
                                                         augmenter_img=augmenter_img,
                                                         augmentations_patch=augmentations_patch,
                                                         augmenter_patch=augmenter_patch,
                                                         augmentation_factor=augmentation_factor)

        elif usage_type == EnumUsage.Segmentation:
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1Segmentation()
        else:
            raise NotImplementedError()

    def __iter__(self, dataset):
        return self.attr_dataset.__iter__(dataset)


if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name=EnumClassificationDataset.ClassificationPatch,
                                     usage_type=EnumUsage.Classification, patch_creator=EnumPatchAlgorithm.FixedPx,
                                     grid_size=1000, input_size=256)

    for input, output, transformation_matrix, name in dataset_factory:
        print("done")
