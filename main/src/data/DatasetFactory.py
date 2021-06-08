import json

from main.FolderInfos import FolderInfos
from main.src.data.classification.DataSentinel1ClassificationPatch import DataSentinel1ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import matplotlib.pyplot as plt
import numpy as np
import json


class DatasetFactory:
    def __init__(self,dataset_name="sentinel1", usage_type="classification",patch_creator="fixed_px",patch_padding="no",grid_size=1000,input_size=1000):
        if patch_creator == "fixed_px":
            if patch_padding == "no":
                path = f"{FolderInfos.input_data_folder}images_informations_preprocessed.json"
                with open(path,"r") as fp:
                    dico_infos = json.load(fp)
                self.attr_patch_creator = Patch_creator0(grid_size_px=grid_size,images_informations_preprocessed=dico_infos)
            else:
                raise NotImplementedError(f"{patch_padding} is not implemented")
        else:
            raise NotImplementedError(f"{patch_creator} is not implemented")

        if usage_type == "classification":
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1ClassificationPatch(self.attr_patch_creator,input_size=input_size)
        elif usage_type == "segmentation":
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1Segmentation()
        else:
            raise NotImplementedError()
    def __getitem__(self, id: int):
        input,output = self.attr_dataset.__getitem__(id)
        return input,output
    def __len__(self):
        return self.attr_dataset.__len__()
    def save_stats(self):
        reso_x_stats = self.attr_dataset.attrend_resolutionX_stats
        reso_y_stats = self.attr_dataset.attrend_resolutionX_stats
        with open(FolderInfos.base_folder+"stats_resoXY.json", "w") as fp:
            json.dump({"resolutionX":reso_x_stats,"resolutionY":reso_y_stats},fp)

if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1",usage_type="classification",patch_creator="fixed_px",patch_padding="no",grid_size=1000,input_size=256)
    length = len(dataset_factory)
    print(f"{length} items in this dataset")
    for id in range(length):
        input,output=dataset_factory[id]
        if id % int(length*0.1) == 0:
            print(f"{id/length*100:.2f} done")
    end = 0