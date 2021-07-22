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
                del original_img
                if device is not None:
                    transfert_to_pytorch = lambda x:torch.Tensor(x).to(device)
                    transfert_from_pytorch = lambda x:x.cpu().detach().numpy()
                else:
                    transfert_to_pytorch = lambda x:torch.Tensor(x)
                    transfert_from_pytorch = lambda x:x.detach().numpy()

                for name,patch,annotation_patch in zip(cache_img.keys(),cache_img.values(),cache_annotations.values()):
                    patch = np.array(patch,dtype=np.float32)
                    annotation_patch = np.array(annotation_patch,dtype=np.float32)
                    transformation_matrix = np.array(dico_infos[name]["transformation_matrix"],dtype=np.float32)
                    patch_full_shape = cv2.warpAffine(reconstructed_image,np.linalg.inv(transformation_matrix)[:-1,:],dsize=reconstructed_image.shape[::-1],flags=cv2.INTER_LANCZOS4)

                    reconstructed_image += patch_full_shape
                    del patch_full_shape
                    with torch.no_grad():
                        patch = patch.reshape((1,*patch.shape))
                        model.eval()
                        prediction: np.ndarray = transfert_from_pytorch(
                            model(transfert_to_pytorch(
                                self.standardizer.standardize(patch)
                            ))
                        )
                        prediction = prediction.flatten()
                        annotation_patch = annotation_patch.flatten()
                        if len(prediction) > 3 or len(annotation_patch) > 3:
                            raise Exception("Cannot show rgb overlay with more than 3 classes")
                        overlay_patch_pred = np.ones((*patch.shape[-2:],3),dtype=np.float32)
                        overlay_patch_true = np.ones((*patch.shape[-2:], 3), dtype=np.float32)
                        for i in range(len(prediction)):
                            overlay_patch_pred[:,:,i] *= prediction[i]
                            overlay_patch_true[:, :, i] *= annotation_patch[i]
                        overlay_full_shape_pred = cv2.warpAffine(overlay_patch_pred, np.linalg.inv(transformation_matrix)[:-1,:],
                                                          dsize=reconstructed_image.shape[::-1], flags=cv2.INTER_LANCZOS4)
                        overlay_full_shape_true = cv2.warpAffine(overlay_patch_true, np.linalg.inv(transformation_matrix)[:-1,:],
                                                          dsize=reconstructed_image.shape[::-1], flags=cv2.INTER_LANCZOS4)
                        overlay_pred += overlay_full_shape_pred
                        overlay_true += overlay_full_shape_true
                        del overlay_full_shape_pred
                        del overlay_full_shape_true
                        del overlay_patch_true
                        del overlay_patch_pred
            return overlay_pred,overlay_true,reconstructed_image
    def normalize(self,image,min=None,max=None):
        if min is None:
            min = np.min(image)
        if max is None:
            max = np.max(image)
        if max == min == 0:
            return image
        if max == min:
            return image/max
        return (image-min)/(max-min)
    def visualize_overlays(self,overlay_pred,overlay_true,reconstructed_image, threshold:int=0.5):

        reconstructed_image = self.normalize(reconstructed_image) * threshold
        [overlay_true,overlay_pred] = [self.normalize(overlay) * (1-threshold) for overlay in [overlay_true,overlay_pred]]
        overlay_true = np.stack((reconstructed_image,)*3,axis=-1)+overlay_true
        overlay_pred = np.stack((reconstructed_image,)*3,axis=-1)+overlay_pred
        import matplotlib
        matplotlib.use("agg")
        plt.figure(1)
        plt.title(f"Overlay true")
        plt.imshow(overlay_true)
        plt.savefig(FolderInfos.base_filename+"rgb_overlay_true.png")
        del overlay_true
        plt.figure(2)
        plt.title(f"Overlay pred")
        plt.imshow(overlay_pred)
        plt.savefig(FolderInfos.base_filename+"rgb_overlay_pred.png")
        del overlay_pred

    def __call__(self, model,device):
        model.load_state_dict(torch.load(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-21_02h37min54s_\2021-07-21_02h37min54s__model_epoch-14_it-15923.pt"))
        self.visualize_overlays(*self.generate_overlay_matrices(self.name_img,model=model,device=device),threshold=0.5)
