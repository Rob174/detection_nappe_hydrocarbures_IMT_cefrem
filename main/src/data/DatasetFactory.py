from main.FolderInfos import FolderInfos
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.classification.ClassificationPatch1 import ClassificationPatch1
from main.src.data.classification.ClassificationPatch2 import ClassificationPatch2
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import matplotlib.pyplot as plt
import json
import plotly.express as px
import pandas as pd
from main.src.param_savers.BaseClass import BaseClass
import torch
import time


class DatasetFactory(BaseClass,torch.utils.data.IterableDataset):
    """Class managing the dataset creation and access with options of:
    - different dataset possible
    - different patch creator possible

    Args:
        dataset_name: str, name of the dataset to build. Currently supported
        - "classificationpatch", dataset of classification on patches with original classes
        - "classificationpatch1", dataset of classification on patches with less classes than the original dataset
        - "classificationpatch2", dataset of classification on patches merging specified classes together to predict if there is something or not
        usage_type: str,
        - "classification",  dataset to classify patches
        - "segmentation", dataset to segment an image
        patch_creator: str for classification only, class to produce and manage patches:
        - "fixed_px": create the PatchCreator0 class which generate fixed px size patches
        grid_size: int, classification only with fixed_px size. To specify the size of a patch
        input_size: int, size of the image given to the model
        exclusion_policy: str, policy to exclude patches. See ClassificationPatch0
        classes_to_use: str, classes names separated but commas to indicate the classes to use
        balance: str, indicate which class to use to balance (or not) the dataset according to classes repartition (see ClassificationPatch)
        margin: int, additionnal parameter to balance classes, cf doc in ClassificationPatch or in BalanceClasses1
        augmentations_img: opt str, list of augmentations to apply seprated by commas
        augmenter_img: opt str, name of the augmenter to use
        augmentations_patch: opt str, list of augmentations to apply seprated by commas
        augmenter_patch: opt str, name of the augmenter to use
    """
    def __init__(self, dataset_name="classificationpatch", usage_type="classification", patch_creator="fixed_px",
                 grid_size=1000, input_size=1000,exclusion_policy="marginmorethan_1000",classes_to_use="seep,spills",
                 balance="nobalance",
                 augmentations_img="none",augmenter_img="noaugmenter",
                 augmentations_patch="none",augmenter_patch="noaugmenter"):
        self.attr_global_name = "data"
        with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", "r") as fp:
            dico_infos = json.load(fp)
        if patch_creator == "fixed_px":
            self.attr_patch_creator = Patch_creator0(grid_size_px=grid_size,
                                                     images_informations_preprocessed=dico_infos,
                                                     exclusion_policy=exclusion_policy)
        else:
            raise NotImplementedError(f"{patch_creator} is not implemented")

        if usage_type == "classification":
            if dataset_name == "classificationpatch":
                self.attr_dataset = ClassificationPatch(self.attr_patch_creator, input_size=input_size,
                                                        balance=balance,
                                                         augmentations_img=augmentations_img,augmenter_img=augmenter_img,
                                                         augmentations_patch=augmentations_patch,augmenter_patch=augmenter_patch)
            elif dataset_name == "classificationpatch1":
                self.attr_dataset = ClassificationPatch1(self.attr_patch_creator, input_size=input_size,
                                                         classes_to_use=classes_to_use,
                                                         balance=balance,
                                                         augmentations_img=augmentations_img,augmenter_img=augmenter_img,
                                                         augmentations_patch=augmentations_patch,augmenter_patch=augmenter_patch)
            elif dataset_name == "classificationpatch2":
                self.attr_dataset = ClassificationPatch2(self.attr_patch_creator, input_size=input_size,
                                                         classes_to_use=classes_to_use,
                                                         balance=balance,
                                                         augmentations_img=augmentations_img,augmenter_img=augmenter_img,
                                                         augmentations_patch=augmentations_patch,augmenter_patch=augmenter_patch)

        elif usage_type == "segmentation":
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1Segmentation()
        else:
            raise NotImplementedError()
        self.attr_length_dataset = len(self.attr_dataset)
    def __iter__(self):
        return self.attr_dataset.__iter__()



if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1", usage_type="classification", patch_creator="fixed_px",grid_size=1000, input_size=256)
    length = len(dataset_factory)
    print(f"{length} items in this dataset")
    for id in range(length):
        input, output, reject = dataset_factory[id]
        if id % int(length * 0.1) == 0:
            print(f"{id / length * 100:.2f} done")
    dataset_factory.save_stats()
    end = 0
