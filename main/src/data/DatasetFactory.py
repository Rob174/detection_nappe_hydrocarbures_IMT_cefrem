import json

from main.FolderInfos import FolderInfos
from main.src.data.classification.DataSentinel1ClassificationPatch import DataSentinel1ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation


class DatasetFactory:
    def __init__(self,dataset_name="sentinel1", usage_type="classification",patch_creator="fixed_px",patch_padding="no",grid_size=1000):
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
                self.attr_dataset = DataSentinel1ClassificationPatch(self.attr_patch_creator)
        elif usage_type == "segmentation":
            if dataset_name == "sentinel1":
                self.attr_dataset = DataSentinel1Segmentation()
        else:
            raise NotImplementedError()
    def __getitem__(self, id: int):
        input,output = self.attr_dataset.__getitem__(id)
        input = self.re
        return
    def __len__(self):
        return self.attr_dataset.__len__()

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    dataset_factory = DatasetFactory(dataset_name="sentinel1",usage_type="classification",patch_creator="fixed_px",patch_padding="no",grid_size=1000)
    for id in range(len(dataset_factory)):
        input,output=dataset_factory[id]
        print(id,input.shape,output.shape)