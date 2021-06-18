from main.FolderInfos import FolderInfos
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.classification.ClassificationPatch1 import ClassificationPatch1
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import matplotlib.pyplot as plt
import json
import plotly.express as px
import pandas as pd
from main.src.param_savers.BaseClass import BaseClass


class DatasetFactory(BaseClass):
    def __init__(self, dataset_name="classificationpatch", usage_type="classification", patch_creator="fixed_px", grid_size=1000, input_size=1000,exclusion_policy="marginmorethan_1000",classes_to_use="seep,spills"):
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
                self.attr_dataset = ClassificationPatch(self.attr_patch_creator, input_size=input_size)
            elif dataset_name == "classificationpatch1":
                self.attr_dataset = ClassificationPatch1(self.attr_patch_creator, input_size=input_size,classes_to_use=classes_to_use)

        elif usage_type == "segmentation":
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1Segmentation()
        else:
            raise NotImplementedError()
        self.attr_length_dataset = len(self.attr_dataset)

    def __getitem__(self, id: int):
        input, output = self.attr_dataset.__getitem__(id)
        return input, output

    def __len__(self):
        return self.attr_dataset.__len__()
    def __iter__(self):
        return self.generator()
    def generator(self):
        for id in range(self.__len__()):
            input,output = self.__getitem__(id)
            if self.attr_patch_creator.reject is True:
                continue
            else:
                yield input,output
    def save_stats(self):
        reso_x_stats = self.attr_dataset.attrend_resolutionX_stats
        reso_y_stats = self.attr_dataset.attrend_resolutionX_stats
        print(f"Stat reolution saved at {FolderInfos.base_folder}stats_resoXY.json")
        with open(FolderInfos.base_folder + "stats_resoXY.json", "w") as fp:
            json.dump({"resolutionX": reso_x_stats, "resolutionY": reso_y_stats}, fp)


    def process_resolution_stats(self,resolution_dict):
        df = pd.DataFrame({
            "ResolutionX": list(resolution_dict["resolutionX"].keys()),
            "ResolutionY": list(resolution_dict["resolutionY"].keys()),
            "NumberX": list(resolution_dict["resolutionX"].values()),
            "NumberY": list(resolution_dict["resolutionY"].values())
        })
        fig = px.bar(df, x='ResolutionX', y='NumberX',title=f"For a {self.attr_patch_creator.attr_grid_size_px} fixed grid size\nand {self.attr_dataset.attr_resizer.attr_out_size_w} dataset output width")
        fig.write_html(FolderInfos.base_filename + "barplot_resolutionX.html")
        plt.clf()
        fig = px.bar(df, x='ResolutionY', y='NumberY',title=f"For a {self.attr_patch_creator.attr_grid_size_px} fixed grid size\nand {self.attr_dataset.attr_resizer.attr_out_size_w} dataset output width")
        fig.write_html(FolderInfos.base_filename + "barplot_resolutionY.html")


if __name__ == "__main__":
    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1", usage_type="classification", patch_creator="fixed_px",grid_size=1000, input_size=256)
    length = len(dataset_factory)
    print(f"{length} items in this dataset")
    for id in range(length):
        input, output = dataset_factory[id]
        if id % int(length * 0.1) == 0:
            print(f"{id / length * 100:.2f} done")
    dataset_factory.save_stats()
    end = 0
