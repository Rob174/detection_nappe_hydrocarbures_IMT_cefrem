import json
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.classification.Standardizer.AbstractStandardizer import AbstractStandardizer
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses, EnumUsage
from main.src.data.patch_creator.enums import EnumPatchAlgorithm


class RGB_Overlay2:
    def __init__(self, standardizer: AbstractStandardizer):

        with open(FolderInfos.base_filename + "parameters.json", "r") as fp:
            self.parameters = json.load(fp)
        self.name_img = "027481_0319CB_0EB7"
        self.standardizer = standardizer


    def generate_overlay_matrices(self, name_img, model,device=None):
        """In this function we will constitute patch after patch the overlay of the image filling true and prediction empty matrices with corresponding patches

        Args:
            name_img: id / name of the image as shown in images_informations_preprocessed.json (better to vizualize but the "official" keys are in the images_preprocessed.hdf5 file)
            model: a created attr_model with pretrained weights already loaded ⚠️
            blending_factor: the percentage (∈ [0.,1.]) of importance given to the color
            device: a pytorch device (gpu)

        Returns: tuple, with  overlay_true,overlay_pred,original_img as np arrays

        """
        with File(FolderInfos.input_data_folder+"images_preprocessed.hdf5","r") as cache:
            original_img = np.array(cache[name_img],dtype=np.float32)  # radar image input (1 channel only)
        overlay_true = np.zeros((*original_img.shape,3))  # prepare the overlays with 3 channels for colors
        overlay_pred = np.zeros((*original_img.shape,3))

        with File(FolderInfos.input_data_folder+"test_cache_image.hdf5","r") as cache_img:
            with File(FolderInfos.input_data_folder+"test_cache_annotations.hdf5","r") as cache_annotations:
                with open(FolderInfos.input_data_folder+"test_cache_img_infos.json","r") as fp:
                    dico_infos = json.load(fp)
                reconstructed_image = np.zeros(original_img.shape)
                if device is not None:
                    transfert_to_pytorch = lambda x:torch.Tensor(x).to(device)
                    transfert_from_pytorch = lambda x:x.cpu().detach().numpy()
                else:
                    transfert_to_pytorch = lambda x:torch.Tensor(x)
                    transfert_from_pytorch = lambda x:x.detach().numpy()

                for name,patch,annotation_patch in zip(cache_img.keys(),cache_img.values(),cache_annotations.values()):
                    transformation_matrix = np.array(dico_infos[name]["transformation_matrix"],dtype=np.float32)
                    patch_full_shape = cv2.warpAffine(reconstructed_image,np.linalg.inv(transformation_matrix),dsize=reconstructed_image.shape,flags=cv2.INTER_LANCZOS4)
                    reconstructed_image += patch_full_shape
                    del patch_full_shape
                    with torch.no_grad():
                        patch = patch.reshape((1,*patch.shape))
                        prediction: np.ndarray = transfert_from_pytorch(
                            model(transfert_to_pytorch(
                                self.standardizer.standardize(patch)
                            ))
                        )
                        prediction = prediction.flatten()
                        if len(prediction) > 3:
                            raise Exception("Cannot show rgb overlay with more than 3 classes")
                        for overlay_dest in [overlay_pred,overlay_true]:
                            overlay = np.ones((*patch.shape[-2:],3),dtype=np.float32)
                            for i in range(len(prediction)):
                                overlay[:,:,i] *= prediction[i]
                            overlay_full_shape = cv2.warpAffine(overlay, np.linalg.inv(transformation_matrix),
                                                              dsize=reconstructed_image.shape, flags=cv2.INTER_LANCZOS4)
                            overlay_dest += overlay_full_shape
            return overlay_pred,overlay_true,reconstructed_image,original_img
    def normalize(self,image,min=None,max=None):
        if min is None:
            min = np.min(image)
        if max is None:
            max = np.max(image)
        return (image-min)/(max-min)
    def visualize_overlays(self,overlay_pred,overlay_true,reconstructed_image,original_img, threshold:int=0.5):

        reconstructed_image = self.normalize(reconstructed_image) * threshold
        [overlay_true,overlay_pred] = [overlay * (1-threshold) for overlay in [overlay_true,overlay_pred]]
        overlay_true = reconstructed_image+overlay_true
        overlay_pred = reconstructed_image+overlay_pred

        plt.figure(1)
        plt.title(f"Overlay true")
        plt.imshow(overlay_true)

        plt.figure(2)
        plt.title(f"Overlay pred")
        plt.imshow(overlay_pred)

        plt.figure(3)
        plt.title(f"Original image")
        plt.imshow(original_img,cmap="gray")

        plt.show()
    def __call__(self, model,device):
        self.generate_overlay_matrices(self.name_img,model=model,device=device)
