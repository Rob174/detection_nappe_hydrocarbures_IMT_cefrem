import torch
import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.models.ModelFactory import ModelFactory, EnumModels, EnumFreeze
from main.src.parsers.Parser0 import Parser0
from main.src.training.TrValidSplit import trvalidsplit

if __name__ == '__main__':
    FolderInfos.init(test_without_data=True)
    model = ModelFactory(model_name=EnumModels.Resnet152, num_classes=2, freeze=EnumFreeze.NoFreeze)
    parser = Parser0()
    arguments = parser()
    dataset = DatasetFactory(dataset_name=arguments.attr_dataset,
                             usage_type=arguments.usage_type,
                             patch_creator=arguments.patch,
                             grid_size=arguments.grid_size,
                             input_size=arguments.input_size,
                             exclusion_policy=arguments.patch_exclude_policy,
                             exclusion_policy_threshold=arguments.patch_exclude_policy_threshold,
                             classes_to_use=arguments.classes,
                             balance=arguments.balance,
                             augmenter_img=arguments.augmenter_img,
                             augmentations_img=arguments.augmentations_img,
                             augmenter_patch=arguments.augmenter_patch,
                             augmentations_patch=arguments.augmentations_patch,
                             augmentation_factor=arguments.augmentation_factor,
                             other_class_adder=arguments.other_class_adder,
                             interval=arguments.interval,
                             )
    [dataset_tr, dataset_valid] = trvalidsplit(dataset)
    device = torch.device("cuda")
    model = model.model.to(device)
    model.load_state_dict(torch.load(
        r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-21_02h37min54s_\2021-07-21_02h37min54s__model_epoch-14_it-15923.pt"))

    for i, [input, output, transformation_matrix, item] in enumerate(dataset_tr):
            if i % 100 == 0:
                print("step ",i)
            img = input.reshape((1,*input.shape))

            with torch.no_grad():
                input_gpu = torch.Tensor(img).float().to(device)
                prediction = model(input_gpu.float())
                del input_gpu
                prediction_npy: np.ndarray = prediction.cpu().detach().numpy()
                if len(prediction_npy[prediction_npy > 0.5]) > 0:
                    print("found ",item,i)

