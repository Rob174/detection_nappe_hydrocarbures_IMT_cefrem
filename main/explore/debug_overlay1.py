import torch
import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.data.Standardizer.StandardizerCacheMixed import StandardizerCacheMixed
from main.src.data.Annotations.NumpyAnnotations import NumpyAnnotations
from main.src.models.ModelFactory import ModelFactory, EnumModels, EnumFreeze
from main.src.parsers.Parser0 import Parser0
from main.src.training.TrValidSplit import trvalidsplit

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    model = ModelFactory(model_name=EnumModels.Resnet152,num_classes=2,freeze=EnumFreeze.NoFreeze)
    model.model.load_state_dict(torch.load(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-21_02h37min54s_\2021-07-21_02h37min54s__model_epoch-14_it-15923.pt"))

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
                             interval=arguments.interval,force_classifpatch=True
                             )
    [dataset_tr, dataset_valid] = trvalidsplit(dataset)
    device = torch.device("cuda")
    batch_valid_input = []
    batch_valid_pred_true = []
    valid_batch_size = 100
    model = model.model.to(device)
    dataset.attr_dataset.set_standardizer(StandardizerCacheMixed(interval=1))
    dataset.attr_dataset.set_annotator(NumpyAnnotations())
    # V2
    # patches = dataset.attr_dataset.make_patches_of_image("027481_0319CB_0EB7")
    # resizer = Resizer(out_size_w=256,interpolation=cv2.INTER_LANCZOS4)
    # lindex_with_smth = []
    # for i,[patch,classif,reject] in enumerate(patches):
    #     patch = resizer.call(patch)
    #     patch = np.stack((patch,)*3,axis=0)
    #     input_npy = patch.reshape((1,*patch.shape))
    #     output_npy = classif.reshape((1,*classif.shape))
    #     with torch.no_grad():
    #         input_gpu = torch.Tensor(input_npy).float().to(device)
    #         prediction = model(input_gpu.float())
    #         del input_gpu
    #         prediction_npy: torch.Tensor = prediction.cpu().detach().numpy()
    #         print(f"true : {output_npy} ; pred : {prediction_npy}")
    #
    #         if 1 in output_npy[0].tolist():
    #             lindex_with_smth.append(i)
    # print("indexes with something : ",lindex_with_smth)

    # V3
    patches = dataset.attr_dataset.make_patches_of_image3("027481_0319CB_0EB7")
    for i,[patch,classif] in enumerate(patches):
        # if i in [94,96,97,151]:
        #     b=0
        patch = np.stack((patch,)*3,axis=0)
        input_npy = patch.reshape((1,*patch.shape))
        output_npy = classif.reshape((1,*classif.shape))
        with torch.no_grad():
            input_gpu = torch.Tensor(input_npy).float().to(device)
            prediction = model(input_gpu.float())
            del input_gpu
            prediction_npy: torch.Tensor = prediction.cpu().detach().numpy()
            print(f"true : {output_npy} ; pred : {prediction_npy}")
            if 1 in classif.flatten().tolist():
                b=0
    # V1
    # for [input, output, transformation_matrix, item] in dataset_valid:
    #     batch_valid_input.append(input)
    #     batch_valid_pred_true.append(output)
    #     if len(batch_valid_input) == valid_batch_size:
    #         input_npy = np.stack(batch_valid_input, axis=0)
    #         output_npy = np.stack(batch_valid_pred_true, axis=0)
    #         batch_valid_input = []
    #         batch_valid_pred_true = []
    #         with torch.no_grad():
    #             input_gpu = torch.Tensor(input_npy).float().to(device)
    #             prediction = model(input_gpu.float())
    #             del input_gpu
    #             output_gpu = torch.Tensor(output_npy).float().to(device)
    #             prediction_npy: torch.Tensor = prediction.cpu().detach().numpy()
    #             b = 0
    #     else:
    #         continue

