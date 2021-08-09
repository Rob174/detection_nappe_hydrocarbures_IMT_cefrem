"""Class managing the attr_dataset creation and access with different type of dataset possible
"""

from typing import Tuple, Dict, Optional

import torch

from main.src.data.Datasets.AbstractDataset import AbstractDataset
from main.src.data.Datasets.HDF5Dataset import HDF5Dataset
from main.src.data.Generators.ClassificationGeneratorCache import ClassificationGeneratorCache
from main.src.data.Generators.ClassificationGeneratorPatch import ClassificationGeneratorPatch
from main.src.enums import EnumAugmenter, EnumBalance, EnumLabelModifier, EnumClassPatchAdder, EnumUsage, EnumClasses, \
    EnumPatchExcludePolicy
from main.src.param_savers.BaseClass import BaseClass


class DatasetFactory(BaseClass, torch.utils.data.IterableDataset):
    """Class managing the attr_dataset creation and access with different type of dataset possible

    Args:
        dataset_name: EnumLabelModifier,
        usage_type: EnumUsage,
        grid_size: int, Generators only with fixed_px size. To specify the size of a patch
        input_size: int, size of the image given to the attr_model
        exclusion_policy: EnumPatchExcludePolicy, policy to exclude patches. See [NoLabelModifier](./Generators/NoLabelModifier.html)
        exclusion_policy_threshold: int, parameter for EnumPatchExcludePolicy.MarginMoreThan
        classes_to_use: Tuple[EnumClasses], the classes to use
        balance: EnumBalance,
        augmentations_img: opt str, list of augmentations to apply seprated by commas
        augmenter_img: opt EnumAugmenter,
        augmentation_factor: int, the number of times that the source image is augmented
        force_classifpatch: bool, force to use the class classificattionpatch
        other_class_adder: EnumClassPatchAdder to select classadder object
        interval: int, interval between two un annotated patches
        choose_dataset, Optional[str] "cache" to use the ClassificationCache method "patch"
    """

    def __init__(self,
                 dataset_name: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 usage_type: EnumUsage = EnumUsage.Classification,
                 grid_size: int = 1000,
                 input_size: int = 1000,
                 exclusion_policy=EnumPatchExcludePolicy.MarginMoreThan,
                 exclusion_policy_threshold: int = 1000,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill),
                 balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img: str = "none",
                 augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor=1, force_classifpatch=False,
                 other_class_adder: EnumClassPatchAdder = EnumClassPatchAdder.NoClassPatchAdder,
                 interval: int = 1,
                 choose_dataset: Optional[str] = None,
                 tr_batch_size: int = 10,
                 valid_batch_size: int = 100

                 ):
        self.attr_global_name = "data"

        if usage_type == EnumUsage.Classification:
            if (
                    input_size == 256 and balance == EnumBalance.BalanceClasses1 and augmenter_img == EnumAugmenter.Augmenter1
                    and augmentations_img == "combinedRotResizeMir_10_0.25_4" and
                    exclusion_policy == EnumPatchExcludePolicy.MarginMoreThan and exclusion_policy_threshold == 10
                    and grid_size == 1000 and not force_classifpatch or choose_dataset == "cache") and choose_dataset != "patch":
                self.attr_dataset = ClassificationGeneratorCache(label_modifier=dataset_name,
                                                                 classes_to_use=classes_to_use,
                                                                 other_class_adder=other_class_adder, interval=interval,
                                                                 tr_batch_size=tr_batch_size,
                                                                 valid_batch_size=valid_batch_size)
            else:
                self.attr_dataset = ClassificationGeneratorPatch(input_size=input_size,
                                                                 classes_to_use=classes_to_use,
                                                                 balance=balance,
                                                                 augmentations_img=augmentations_img,
                                                                 augmenter_img=augmenter_img,
                                                                 augmentation_factor=augmentation_factor,
                                                                 label_modifier=dataset_name,
                                                                 grid_size_px=grid_size,
                                                                 threshold_margin=exclusion_policy_threshold,
                                                                 tr_batch_size=tr_batch_size,
                                                                 valid_batch_size=valid_batch_size
                                                                 )


        elif usage_type == EnumUsage.Segmentation:
            if dataset_name == "sentinel1":
                raise NotImplementedError()
        else:
            raise NotImplementedError()

    def __iter__(self, dataset):
        return self.attr_dataset.__iter__(dataset)

    def len(self, dataset):
        return self.attr_dataset.len(dataset)

    def set_datasets(self, image_dataset: HDF5Dataset, label_dataset: AbstractDataset, dico_infos: Dict):
        """Change the origin of the patches

        Args:
            image_dataset: HDF5Dataset
            label_dataset: AbstractDataset Points or Images
            dico_infos: Dict, containing for each id of image the source image (under key source_img) and the transformation matrix (under key transformation_matrix) applied to get the patch
        Returns:

        """
        self.attr_dataset.set_datasets(image_dataset, label_dataset, dico_infos)
