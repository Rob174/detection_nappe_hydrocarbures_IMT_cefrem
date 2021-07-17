import numpy as np
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses, EnumUsage
from main.src.data.patch_creator.enums import EnumPatchAlgorithm


class RGB_Overlay_Patch:
    def __init__(self, dataset_name=EnumLabelModifier.LabelModifier1, usage_type=EnumUsage.Classification,
                 patch_creator=EnumPatchAlgorithm.FixedPx,
                 grid_size=1000, input_size=256,
                 classes_to_use=(EnumClasses.Other, EnumClasses.Seep, EnumClasses.Spill)):
        self.dataset = DatasetFactory(dataset_name=dataset_name, usage_type=usage_type, patch_creator=patch_creator,
                                      grid_size=grid_size, input_size=input_size, classes_to_use=classes_to_use,
                                      force_classifpatch=True)

    def __call__(self, name_img, model, blending_factor=0.5, device=None,threshold=False):
        return self.call(name_img, model, blending_factor, device,threshold)

    def call(self, name_img, model, blending_factor=0.25, device=None, threshold=False):
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
        with progress:  # progress bar manager to use the progress bar
            progression = progress.add_task("generation", name="[red]Progress", total=len(patches))
            skipped = 0
            for id, [input, output, filter] in enumerate(patches):
                if filter is True:  # skip the image if the attr_dataset ask to skip this patch (can be for multiple reasons -> see DatasetFactory parameters supplied)
                    skipped += 1
                    continue
                input_adapted = np.stack((input, input, input), axis=0)  # convert patch to rgb
                # pytorch can only make predictions for batches of images. That is why we create a "batch" of one image by adding one dimension to the image:
                # shape : (1, img_width, img_height, 3)
                input_adapted = input_adapted.reshape((1, *input_adapted.shape))
                # predict output with cpu if no device (gpu) is provided else predict with gpu and ransfer the result on cpu

                with torch.no_grad():
                    prediction = model(input) if device is None else model(
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
                progress.update(progression, advance=1)
        print(f"skipped {skipped}")
        return overlay_true, overlay_pred, original_img


if __name__ == "__main__":
    choice_folder1 = '2021-07-17_18h40min00s_'
    from main.src.models.ModelFactory import ModelFactory
    import json, os

    name = "027481_0319CB_0EB7"
    FolderInfos.init(test_without_data=True)
    folder = FolderInfos.data_folder + choice_folder1 + FolderInfos.separator
    epoch = 11
    iteration = 10615

    if os.path.exists(f"{folder}{choice_folder1}_{name}_it_{iteration}_epoch_{epoch}_rgb_overlay_pred.png") is True:
        print("loading from cache")
        import matplotlib.pyplot as plt
        from PIL import Image

        image_true = Image.open(f"{folder}{choice_folder1}_{name}_it_{iteration}_epoch_{epoch}_rgb_overlay_true.png")
        image_pred = Image.open(f"{folder}{choice_folder1}_{name}_it_{iteration}_epoch_{epoch}_rgb_overlay_pred.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(image_true)
        plt.figure(figsize=(10, 10))
        plt.imshow(image_pred)
        plt.show()
    else:
        print("not already computed, processing...")
        with open(folder + choice_folder1 + "parameters.json", "r") as fp:
            dico = json.load(fp)
        classes_to_use = [EnumClasses.Seep, EnumClasses.Spill]
        rgb_overlay = RGB_Overlay_Patch(dataset_name=EnumLabelModifier.LabelModifier1,
                                        usage_type=EnumUsage.Classification,
                                        patch_creator=EnumPatchAlgorithm.FixedPx,
                                        grid_size=1000,
                                        input_size=256,
                                        classes_to_use=classes_to_use
                                        )

        import torch

        device = torch.device("cuda")
        model = ModelFactory(model_name=dico["trainer"]["attr_model"]["attr_model_name"],
                             num_classes=dico["trainer"]["attr_model"]["attr_num_classes"])()
        model.to(device)
        model.load_state_dict(torch.load(f"{folder}{choice_folder1}_model_epoch-{epoch}_it-{iteration}.pt"))
        array_overlay = rgb_overlay(name_img=name, model=model, blending_factor=0.5, device=device, threshold=True)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[0])
        plt.gcf().text(0.02, 0.75, f"RGB with order {classes_to_use}", fontsize=14)
        plt.savefig(f"{folder}{choice_folder1}_{name}_it_{iteration}_epoch_{epoch}_rgb_overlay_true.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[1])
        plt.gcf().text(0.02, 0.75, f"RGB with order {classes_to_use}", fontsize=14)
        plt.savefig(f"{folder}{choice_folder1}_{name}_it_{iteration}_epoch_{epoch}_rgb_overlay_pred.png")
        plt.figure(figsize=(10, 10))
        plt.imshow(array_overlay[2], cmap="gray")
