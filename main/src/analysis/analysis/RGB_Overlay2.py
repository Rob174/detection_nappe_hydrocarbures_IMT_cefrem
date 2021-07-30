import json
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricPreprocessedCache import FabricPreprocessedCache
from main.src.data.Datasets.Fabrics.FabricTestCache import FabricTestCache
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.models.ModelFactory import ModelFactory, EnumClasses
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback

matplotlib.use("agg")


class RGB_Overlay2(AbstractCallback,BaseClass):
    def __init__(self, standardizer: AbstractStandardizer, model: ModelFactory, device: Optional = None):
        super(RGB_Overlay2, self).__init__()
        self.name_img = "027481_0319CB_0EB7"
        self.standardizer = standardizer
        self.model = model.model
        self.device = device
        self.attr_name = self.__class__.__name__

    def generate_overlay_matrices(self, name_img, threshold: float = 0.5):
        """In this function we will constitute patch after patch the overlay of the image filling true and prediction empty matrices with corresponding patches

        Args:
            name_img: id / name of the image as shown in images_informations_preprocessed.json (better to vizualize but the "official" keys are in the images_preprocessed.hdf5 file)
            model: a created attr_model with pretrained weights already loaded ⚠️
            blending_factor: the percentage (∈ [0.,1.]) of importance given to the color
            device: a pytorch device (gpu)

        Returns: tuple, with  overlay_true,overlay_pred,original_img as np arrays

        """
        images,annotations,infos = FabricTestCache()()
        original_img,original_annot,_ = FabricPreprocessedCache()()
        original_image = np.array(original_img.get(name_img), dtype=np.float16)
        original_shape = original_image.shape  # radar image input (1 channel only)

        if self.device is not None:
            print("using gpu")
            transfert_to_pytorch = lambda x: torch.Tensor(x).to(self.device)
            transfert_from_pytorch = lambda x: x.cpu().detach().numpy()
            self.model.to(self.device)
        else:
            transfert_to_pytorch = lambda x: torch.Tensor(x)
            transfert_from_pytorch = lambda x: x.detach().numpy()
        # True label visualization + reconstructed image
        overlay_true = np.ones((*original_shape, 3), dtype=np.float16)
        modifier = LabelModifier1(original_class_mapping=annotations.attr_mapping,
                                              classes_to_use=(EnumClasses.Seep,EnumClasses.Spill))
        for name in (images.keys()):
            patch = images.get(name)
            annotation_patch = annotations.get(name)
            annotation_patch = modifier.make_classification_label(annotation_patch)
            print(name, end="\r")
            inverse_transformation_matrix = np.linalg.inv(
                np.array(infos[name]["transformation_matrix"], dtype=np.float32))

            patch_shape = patch.shape[-2:]
            # compute manually points to slice
            corner_upleft = np.array([0, 0, 1])
            mapped_corner_upperleft = np.round(inverse_transformation_matrix.dot(corner_upleft)).astype(
                np.int32).tolist()
            corner_bottomright = np.array([*patch_shape[::-1], 1])
            mapped_corner_bottomright = np.round(inverse_transformation_matrix
                                                 .dot(corner_bottomright)).astype(np.int32).tolist()

            if len(annotation_patch) > 3:
                raise Exception("Cannot show rgb overlay with more than 3 classes")

            annotation_patch = np.array(annotation_patch, dtype=np.float16).flatten()
            for i in range(len(annotation_patch)):
                overlay_true[mapped_corner_upperleft[1]:mapped_corner_bottomright[1],
                mapped_corner_upperleft[0]:mapped_corner_bottomright[0], i] *= annotation_patch[i]
        reconstructed_image = self.normalize(original_image) * threshold
        reconstructed_image = np.stack((reconstructed_image,) * 3, axis=-1)
        overlay_true = reconstructed_image + self.normalize(overlay_true) * (1 - threshold)

        name = "true"
        plt.figure(1)
        plt.title(f"Overlay " + name)
        plt.imshow(np.array(overlay_true * 255, dtype=np.uint8))
        plt.savefig(FolderInfos.base_filename + f"rgb_overlay_{name}.png")
        del overlay_true

        overlay_pred = np.ones((*original_shape, 3), dtype=np.float16)
        for name, patch in zip(images.keys(), images.values()):
            print(name, end="\r")
            inverse_transformation_matrix = np.linalg.inv(
                np.array(infos[name]["transformation_matrix"], dtype=np.float32))
            patch_shape = patch.shape[-2:]
            patch: np.ndarray = np.stack((patch,)*3,axis=0)
            patch: np.ndarray = np.reshape(patch, (1, *patch.shape))
            with torch.no_grad():
                self.model.eval()
                prediction: np.ndarray = transfert_from_pytorch(
                    self.model(transfert_to_pytorch(
                        np.array(self.standardizer.standardize(patch), dtype=np.float32)
                    ))
                )
                del patch

                prediction = prediction.flatten()
                if len(prediction) > 3:
                    raise Exception("Cannot show rgb overlay with more than 3 classes")
                # compute manually points to slice
                corner_upleft = np.array([0, 0, 1])
                mapped_corner_upperleft = np.round(inverse_transformation_matrix.dot(corner_upleft)).astype(
                    np.int32).tolist()
                corner_bottomright = np.array([*patch_shape[::-1], 1])
                mapped_corner_bottomright = np.round(
                    inverse_transformation_matrix.dot(corner_bottomright)).astype(np.int32).tolist()

                for i in range(len(prediction)):
                    overlay_pred[mapped_corner_upperleft[1]:mapped_corner_bottomright[1],
                    mapped_corner_upperleft[0]:mapped_corner_bottomright[0], i] *= 1 if prediction[
                                                                                            i] > 0.5 else 0

        overlay_pred = reconstructed_image + self.normalize(overlay_pred) * (1 - threshold)
        name = "pred"
        plt.figure(2)
        plt.title(f"Overlay " + name)
        plt.imshow(np.array(overlay_pred * 255, dtype=np.uint8))
        plt.savefig(FolderInfos.base_filename + f"rgb_overlay_{name}.png")
        del overlay_pred

    def normalize(self, image, min=None, max=None):
        if min is None:
            min = np.min(image)
        if max is None:
            max = np.max(image)
        if max == min == 0:
            return image
        if max == min:
            return image / max
        return (image - min) / (max - min)

    def __call__(self):
        self.generate_overlay_matrices(self.name_img, threshold=0.5)

    def on_end(self):
        self.generate_overlay_matrices(self.name_img, threshold=0.5)

if __name__ == '__main__':
    from main.src.models.ModelFactory import ModelFactory
    from main.src.enums import *
    # from main.src.analysis.analysis.RGB_Overlay2 import RGB_Overlay2
    from main.src.data.Standardizer.StandardizerCacheMixed import StandardizerCacheMixed
    import torch
    from main.FolderInfos import FolderInfos

    id = "2021-07-29_23h56min55s"
    path_pt_file = r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-29_23h56min55s_\2021-07-29_23h56min55s__model_epoch-41_it-3510.pt"
    FolderInfos.init(with_id=id)
    model = ModelFactory(EnumModels.Resnet152, num_classes=2, freeze=EnumFreeze.NoFreeze)
    device = torch.device("cuda")
    model.model.load_state_dict(torch.load(path_pt_file))
    model.model.eval()
    rgb_overlay = RGB_Overlay2(standardizer=StandardizerCacheMixed(interval=1),
                               model=model, device=device)
    rgb_overlay()