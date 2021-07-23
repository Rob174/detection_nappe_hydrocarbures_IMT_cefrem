from main.src.models.ModelFactory import ModelFactory
from main.src.enums import *
from main.src.analysis.analysis.RGB_Overlay2 import RGB_Overlay2
from main.src.data.classification.Standardizer.StandardizerCacheMixed import StandardizerCacheMixed
import torch
from main.FolderInfos import FolderInfos

if __name__ == '__main__':

    id = "2021-07-22_16h46min16s"
    path_pt_file = r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-22_16h46min16s_\2021-07-22_16h46min16s__model_epoch-9_it-15923.pt"
    FolderInfos.init(with_id=id)
    model = ModelFactory(EnumModels.Resnet152, num_classes=2, freeze=EnumFreeze.NoFreeze).model
    device = torch.device("cuda")
    model.load_state_dict(torch.load(path_pt_file))
    model.eval()
    rgb_overlay = RGB_Overlay2(standardizer=StandardizerCacheMixed(interval=1))
    rgb_overlay(model, device)