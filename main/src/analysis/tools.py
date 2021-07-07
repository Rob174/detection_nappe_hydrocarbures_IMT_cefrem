import cv2
import numpy as np
import torch
import rich
import matplotlib.pyplot as plt
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses, EnumUsage
from main.src.data.patch_creator.enums import EnumPatchAlgorithm


class RGB_Overlay_Patch:
    def __init__(self,dataset_name=EnumLabelModifier.LabelModifier1,usage_type=EnumUsage.Classification,patch_creator=EnumPatchAlgorithm.FixedPx,
                 grid_size=1000,input_size=256,classes_to_use=(EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
        self.dataset = DatasetFactory(dataset_name=dataset_name, usage_type=usage_type, patch_creator=patch_creator,
                                      grid_size=grid_size , input_size=input_size,classes_to_use=classes_to_use,force_classifpatch=True)

    def __call__(self,name_img,model,blending_factor=0.5,device=None):
        return self.call(name_img,model,blending_factor,device)
    def call(self,name_img,model,blending_factor=0.25,device=None):
        """In this function we will constitute patch after patch the overlay of the image filling true and prediction empty matrices with corresponding patches

        Args:
            name_img: id / name of the image as shown in images_informations_preprocessed.json (better to vizualize but the "official" keys are in the images_preprocessed.hdf5 file)
            model: a created model with pretrained weights already loaded ⚠️
            blending_factor: the percentage (∈ [0.,1.]) of importance given to the color
            device: a pytorch device (gpu)

        Returns: tuple, with  overlay_true,overlay_pred,original_img as np arrays

        """
        original_img = np.array(self.dataset.attr_dataset.images[name_img]) # radar image input (1 channel only)
        overlay_true = np.zeros((*original_img.shape,3),dtype=np.float32) # prepare the overlays with 3 channels for colors
        overlay_pred = np.zeros((*original_img.shape,3),dtype=np.float32)
        normalize = lambda x:(x-np.min(original_img))/(np.max(original_img)-np.min(original_img))
        # get the list of patches organized as follows [[input_patch0,output_patch0,filter_patch0],....
        patches = self.dataset.attr_dataset.make_patches_of_image(name_img)

        progress = Progress(
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()
        ) # prepare the progress bar -> tell which columns to display
        with progress: # progress bar manager to use the progress bar
            progression = progress.add_task("generation", name="[red]Progress", total=len(patches))
            skipped = 0
            for id,[input,output,filter] in enumerate(patches):
                if filter is True: # skip the image if the dataset ask to skip this patch (can be for multiple reasons -> see DatasetFactory parameters supplied)
                    skipped += 1
                    continue
                input_adapted = np.stack((input,input,input),axis=0) # convert patch to rgb
                # pytorch can only make predictions for batches of images. That is why we create a "batch" of one image by adding one dimension to the image:
                #shape : (1, img_width, img_height, 3)
                input_adapted = input_adapted.reshape((1,*input_adapted.shape))
                # predict output with cpu if no device (gpu) is provided else predict with gpu and ransfer the result on cpu
                prediction = model(input) if device is None else model(torch.tensor(input_adapted).to(device)).cpu()
                # get the pixel position of the patch
                pos_x,pos_y = self.dataset.attr_patch_creator.get_position_patch(id,original_img.shape)
                if len(output) > 3:
                    raise Exception(f"{len(output)} classes : Too much : cannot use this vizualization technic")
                input = normalize(input)
                resized_grid_size = self.dataset.attr_patch_creator.attr_grid_size_px
                input = np.stack((input,)*3,axis=-1) # convert in rgb. NB: (input,)*3 <=> (input,input,input)
                # initialize the overlay for the patch
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
        print(f"skipped {skipped}")
        return overlay_true,overlay_pred,original_img

if __name__ == "__main__":
    choice_folder1 = '2021-07-07_01h47min15s_'
    from main.src.models.ModelFactory import ModelFactory
    import json, os

    name = "027481_0319CB_0EB7"
    FolderInfos.init(test_without_data=True)
    folder = FolderInfos.data_folder + choice_folder1 + FolderInfos.separator

    if os.path.exists(folder + choice_folder1 + "_" + name + "_rgb_overlay_pred.png") is True:
        print("loading from cache")
        import matplotlib.pyplot as plt
        from PIL import Image

        image_true = Image.open(folder + choice_folder1 + "_" + name + "_rgb_overlay_true.png")
        image_pred = Image.open(folder + choice_folder1 + "_" + name + "_rgb_overlay_pred.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_true)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pred)
        plt.show()
    else:
        print("not already computed, processing...")
        with open(folder + choice_folder1 + "parameters.json", "r") as fp:
            dico = json.load(fp)

        rgb_overlay = RGB_Overlay_Patch(dataset_name=EnumLabelModifier.LabelModifier1, usage_type=EnumUsage.Classification,
                                        patch_creator=EnumPatchAlgorithm.FixedPx,
                                        grid_size=1000,
                                        input_size=256,
                                        classes_to_use=[EnumClasses.Seep,EnumClasses.Spill]
                                        )
        epoch = 9
        iteration = 11562
        import torch

        device = torch.device("cuda")
        model = ModelFactory(model_name=dico["model"]["attr_model_name"],
                             num_classes=dico["model"]["attr_num_classes"])()
        model.to(device)
        model.load_state_dict(torch.load(f"{folder}{choice_folder1}_model_epoch-{epoch}_it-{iteration}.pt"))
        array_overlay = rgb_overlay(name_img=name, model=model, blending_factor=0.5, device=device)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[0])
        plt.gcf().text(0.02, 0.75, f"RGB with order {[EnumClasses.Seep,EnumClasses.Spill]}", fontsize=14)
        plt.savefig(f"{folder}{choice_folder1}_{name}_rgb_overlay_true.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[1])
        plt.gcf().text(0.02, 0.75, f"RGB with order {[EnumClasses.Seep,EnumClasses.Spill]}", fontsize=14)
        plt.savefig(f"{folder}{choice_folder1}_{name}_rgb_overlay_pred.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[2], cmap="gray")
        plt.show()