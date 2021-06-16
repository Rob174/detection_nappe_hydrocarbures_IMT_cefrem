import cv2
import numpy as np
import torch
import rich
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory


def moving_mean(x,window):
    new_values = []
    for i in range(len(x)-window):
        new_values.append(np.mean(x[i:min(i+window,len(x))]))
    return new_values

class RGB_Overlay_Patch:
    def __init__(self,dataset_name="sentinel1",usage_type="classification",patch_creator="fixed_px",grid_size=1000,input_size=256):
        FolderInfos.init(test_without_data=False)
        self.dataset = DatasetFactory(dataset_name=dataset_name, usage_type=usage_type, patch_creator=patch_creator,grid_size=grid_size , input_size=input_size)

    def __call__(self,name_img,model,blending_factor=0.5,device=None,resize_factor=4):
        original_img = self.dataset.attr_dataset.images[name_img]
        transformation_matrix = np.array([[1/resize_factor,0,0],
                                          [0,1/resize_factor,0],
                                          [0,0,1]])
        new_shape = list(map(lambda x:int(transformation_matrix.dot(np.array([x,0,1]))[0]),original_img.shape)) + [3]
        overlay_true = np.zeros((*original_img.shape,3),dtype=np.float32)
        overlay_pred = np.zeros((*original_img.shape,3),dtype=np.float32)
        normalize = lambda x:(x-np.min(original_img))/(np.max(original_img)-np.min(original_img))
        patches = self.dataset.attr_dataset.make_patches_of_image(name_img)

        progress = Progress(
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()
        )
        with progress:
            progression = progress.add_task("generation", name="[red]Progress", total=len(patches))
            for id,[input,output,filter] in enumerate(patches):
                if filter is True:
                    print("skipping")
                    continue
                input_adapted = np.stack((input,input,input),axis=0)
                input_adapted = input_adapted.reshape((1,*input_adapted.shape))
                prediction = model(input) if device is None else model(torch.tensor(input_adapted).to(device)).cpu()
                pos_x,pos_y = self.dataset.attr_patch_creator.get_position_patch(id,original_img.shape)
                if len(output) > 3:
                    raise Exception(f"{len(output)} classes : Too much : cannot use this vizualization technic")
                input = normalize(input)
                resized_grid_size = 1000#int(transformation_matrix.dot(np.array([self.dataset.attr_patch_creator.attr_grid_size_px,0,1]))[0])
                # input = cv2.resize(input,dsize=(resized_grid_size,resized_grid_size), interpolation=cv2.INTER_CUBIC)
                input = np.stack((input,)*3,axis=-1) # convert in rgb
                color_true = np.ones((resized_grid_size,resized_grid_size,3))
                color_pred = np.ones((resized_grid_size,resized_grid_size,3))
                for i,c in enumerate(output):
                    color_true[:,:,i] *= c
                if device is not None:
                    prediction = prediction.cpu().detach().numpy()
                for i,c in enumerate(prediction[0]):

                    color_pred[:,:,i] *= c

                coordx1_not_resize = pos_x
                # coordx1 = int(transformation_matrix.dot(np.array([coordx1_not_resize,0,1]))[0])
                coordx2_not_resize = coordx1_not_resize + self.dataset.attr_patch_creator.attr_grid_size_px
                # coordx2 = int(transformation_matrix.dot(np.array([coordx2_not_resize,0,1]))[0])
                coordy1_not_resize = pos_y
                # coordy1 = int(transformation_matrix.dot(np.array([0,coordy1_not_resize,1]))[1])
                coordy2_not_resize = coordy1_not_resize + self.dataset.attr_patch_creator.attr_grid_size_px
                # coordy2 = int(transformation_matrix.dot(np.array([0,coordy2_not_resize,1]))[1])
                overlay_true[coordx1_not_resize:coordx2_not_resize,coordy1_not_resize:coordy2_not_resize,:] = input * (1-blending_factor) + color_true * blending_factor
                overlay_pred[coordx1_not_resize:coordx2_not_resize,coordy1_not_resize:coordy2_not_resize,:] = input * (1-blending_factor) + color_pred * blending_factor
                progress.update(progression, advance=1)
        return overlay_true,overlay_pred

if __name__ == "__main__":
    from main.src.models.ModelFactory import ModelFactory
    import json
    FolderInfos.init(test_without_data=True)
    choice_folder = "2021-06-11_12h30min51s_"
    folder = FolderInfos.data_folder + choice_folder + FolderInfos.separator
    with open(folder + choice_folder + "parameters.json", "r") as fp:
        dico = json.load(fp)

    rgb_overlay = RGB_Overlay_Patch(usage_type="classification", patch_creator="fixed_px",
                                    grid_size=dico["data"]["dataset"]["attr_patch_creator"]["attr_grid_size_px"],
                                    input_size=dico["data"]["dataset"]["attr_dataset"]["attr_resizer"][
                                        "attr_out_size_w"])
    epoch = 0
    iteration = 6080
    import torch

    device = torch.device("cuda")
    model = ModelFactory(model_name=dico["model"]["attr_model_name"], num_classes=dico["model"]["attr_num_classes"])()
    model.to(device)
    model.load_state_dict(torch.load(f"{folder}{choice_folder}_model_epoch-{epoch}_it-{iteration}.pt"))
    array_overlay = rgb_overlay(name_img="027481_0319CB_0EB7", model=model, blending_factor=0.5, device=device)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,10))
    plt.imshow(array_overlay[0])
    plt.figure(figsize=(10,10))
    plt.imshow(array_overlay[1])
    print(array_overlay)