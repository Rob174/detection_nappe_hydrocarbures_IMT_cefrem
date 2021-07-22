import cv2
import matplotlib
import numpy as np
import json, os, torch
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.data.classification.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.enums import EnumLabelModifier
from main.src.enums import EnumClasses, EnumUsage
from main.src.enums import EnumPatchAlgorithm
from main.src.data.resizer import Resizer
from main.src.models.ModelFactory import ModelFactory


class RGB_Overlay_Patch:
    def __init__(self, standardizer: AbstractStandardizer,dataset_name=EnumLabelModifier.LabelModifier1, usage_type=EnumUsage.Classification,
                 patch_creator=EnumPatchAlgorithm.FixedPx,
                 grid_size=1000, input_size=256,
                 classes_to_use=(EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
        self.dataset = DatasetFactory(dataset_name=dataset_name, usage_type=usage_type, patch_creator=patch_creator,
                                      grid_size=grid_size, input_size=input_size, classes_to_use=classes_to_use,
                                      force_classifpatch=True)
        self.classes_to_use = classes_to_use

        with open(FolderInfos.base_filename + "parameters.json", "r") as fp:
            self.parameters = json.load(fp)
        self.name_img = "027481_0319CB_0EB7"
        self.standardizer = standardizer
        self.dataset.attr_dataset.set_standardizer(standardizer=standardizer)

    def __call__(self, epoch: int, iteration: int, model,device,blending_factor: float = 0.5,
             threshold: bool = True, num_classes: int = 2):
        return self.call(FolderInfos.base_folder,epoch, iteration, self.name_img, model,device, blending_factor,
             threshold, num_classes)

    def generate_overlay_matrices(self, name_img, model, blending_factor=0.25, device=None, threshold=False):
        """In this function we will constitute patch after patch the overlay of the image filling true and prediction empty matrices with corresponding patches

        Args:
            name_img: id / name of the image as shown in images_informations_preprocessed.json (better to vizualize but the "official" keys are in the images_preprocessed.hdf5 file)
            model: a created attr_model with pretrained weights already loaded ⚠️
            blending_factor: the percentage (∈ [0.,1.]) of importance given to the color
            device: a pytorch device (gpu)

        Returns: tuple, with  overlay_true,overlay_pred,original_img as np arrays

        """
        def normalize(input):
            max = np.max(input)
            min = np.min(input)
            if max-min == 0:
                return input-min+1
            else:
                return (input-min)/(max-min)
        original_img = normalize(np.array(self.dataset.attr_dataset.images[name_img]))*blending_factor  # radar image input (1 channel only)
        overlay_true = np.stack((original_img,)*3,axis=-1)  # prepare the overlays with 3 channels for colors
        overlay_pred = np.stack((original_img,)*3,axis=-1)
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
        )  # prepare the progress bar -> tell which columns to display
        skipped = 0
        resizer = Resizer(out_size_w=256,interpolation=cv2.INTER_LANCZOS4)
        for id, [input, output, filter] in enumerate(patches):

            if filter is True:  # skip the image if the attr_dataset ask to skip this patch (can be for multiple reasons -> see DatasetFactory parameters supplied)
                skipped += 1
                continue
            input = resizer.call(input)
            input_adapted = np.stack((input, input, input), axis=0)  # convert patch to rgb
            # pytorch can only make predictions for batches of images. That is why we create a "batch" of one image by adding one dimension to the image:
            # shape : (1, 3, img_width, img_height)
            input_adapted = input_adapted.reshape((1, *input_adapted.shape))
            # predict output with cpu if no device (gpu) is provided else predict with gpu and ransfer the result on cpu

            with torch.no_grad():
                prediction = model(input_adapted) if device is None else model(
                    torch.tensor(input_adapted).to(device)).cpu().detach().numpy()
            # get the pixel position of the patch
            pos_x, pos_y = self.dataset.attr_patch_creator.get_position_patch(id, original_img.shape)
            if len(output) > 3:
                raise Exception(f"{len(output)} classes : Too much : cannot use this vizualization technic")
            resized_grid_size = self.dataset.attr_patch_creator.attr_grid_size_px
            # initialize the overlay for the patch
            color_true = np.ones((resized_grid_size, resized_grid_size, 3))
            color_pred = np.ones((resized_grid_size, resized_grid_size, 3))
            for i, c in enumerate(output):
                color_true[:, :, i] *= c
            if threshold is True:
                for i, c in enumerate(prediction[0]):
                    color_pred[:, :, i] *= 0 if c <= 0.5 else 1
            else:
                for i, c in enumerate(prediction[0]):
                    color_pred[:, :, i] *= c
            coordx1_not_resize = pos_x
            coordx2_not_resize = coordx1_not_resize + self.dataset.attr_patch_creator.attr_grid_size_px
            coordy1_not_resize = pos_y
            coordy2_not_resize = coordy1_not_resize + self.dataset.attr_patch_creator.attr_grid_size_px
            overlay_true[coordx1_not_resize:coordx2_not_resize, coordy1_not_resize:coordy2_not_resize,
            :] += color_true * (1-blending_factor)
            overlay_pred[coordx1_not_resize:coordx2_not_resize, coordy1_not_resize:coordy2_not_resize,
            :] += color_pred * (1-blending_factor)
        print(f"skipped {skipped}")
        return overlay_true, overlay_pred, original_img
    def call(self, folder: str,epoch: int, iteration: int, name_img: str, model,device,blending_factor: float = 0.5,
             threshold: bool = True, num_classes: int = 2):
        if os.path.exists(f"{folder}_{name_img}_it_{iteration}_epoch_{epoch}_rgb_overlay_pred.png") is True:
            return
        else:
            matplotlib.use("agg")
            array_overlay = self.generate_overlay_matrices(name_img=name_img, model=model,
                                                           blending_factor=blending_factor,
                                                           device=device, threshold=True)

            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 10))
            plt.imshow(array_overlay[0])
            plt.gcf().text(0.02, 0.75, f"RGB with order {self.classes_to_use}", fontsize=14)
            plt.savefig(f"{folder}_{name_img}_it_{iteration}_epoch_{epoch}_rgb_overlay_true.png")
            plt.figure(figsize=(10, 10))
            plt.imshow(array_overlay[1])
            plt.gcf().text(0.02, 0.75, f"RGB with order {self.classes_to_use}", fontsize=14)
            plt.savefig(f"{folder}_{name_img}_it_{iteration}_epoch_{epoch}_rgb_overlay_pred.png")
