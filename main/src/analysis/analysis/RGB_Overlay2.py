import json
import torch

import matplotlib.pyplot as plt
import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.Standardizer.AbstractStandardizer import AbstractStandardizer
import matplotlib
matplotlib.use("agg")


class RGB_Overlay2:
    def __init__(self, standardizer: AbstractStandardizer):

        with open(FolderInfos.base_filename + "parameters.json", "r") as fp:
            self.parameters = json.load(fp)
        self.name_img = "027481_0319CB_0EB7"
        self.standardizer = standardizer


    def generate_overlay_matrices(self, name_img, model,device=None, threshold: float = 0.5):
        """In this function we will constitute patch after patch the overlay of the image filling true and prediction empty matrices with corresponding patches

        Args:
            name_img: id / name of the image as shown in images_informations_preprocessed.json (better to vizualize but the "official" keys are in the images_preprocessed.hdf5 file)
            model: a created attr_model with pretrained weights already loaded ⚠️
            blending_factor: the percentage (∈ [0.,1.]) of importance given to the color
            device: a pytorch device (gpu)

        Returns: tuple, with  overlay_true,overlay_pred,original_img as np arrays

        """
        with File(FolderInfos.input_data_folder+"images_preprocessed.hdf5","r") as cache:
            original_image = np.array(cache[name_img],dtype=np.float16)
            original_shape = cache[name_img].shape  # radar image input (1 channel only)

        with File(FolderInfos.input_data_folder+"test_cache_image.hdf5","r") as cache_img:
            with File(FolderInfos.input_data_folder+"test_cache_annotations.hdf5","r") as cache_annotations:
                with open(FolderInfos.input_data_folder+"test_cache_img_infos.json","r") as fp:
                    dico_infos = json.load(fp)
                if device is not None:
                    print("using gpu")
                    transfert_to_pytorch = lambda x:torch.Tensor(x).to(device)
                    transfert_from_pytorch = lambda x:x.cpu().detach().numpy()
                    model.to(device)
                else:
                    transfert_to_pytorch = lambda x:torch.Tensor(x)
                    transfert_from_pytorch = lambda x:x.detach().numpy()
                # True label visualization + reconstructed image
                overlay_true = np.ones((*original_shape,3),dtype=np.float16)

                for name, patch,annotation_patch in zip(cache_img.keys(),cache_img.values(),cache_annotations.values()):
                    print(name,end="\r")
                    inverse_transformation_matrix = np.linalg.inv(np.array(dico_infos[name]["transformation_matrix"], dtype=np.float32))

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
                reconstructed_image = np.stack((reconstructed_image,)*3,axis=-1)
                overlay_true = reconstructed_image+self.normalize(overlay_true) * (1-threshold)

                name = "true"
                plt.figure(1)
                plt.title(f"Overlay " + name)
                plt.imshow(np.array(overlay_true*255,dtype=np.uint8))
                plt.savefig(FolderInfos.base_filename + f"rgb_overlay_{name}.png")
                del overlay_true

                overlay_pred = np.ones((*original_shape, 3), dtype=np.float16)
                for name,patch in zip(cache_img.keys(),cache_img.values()):
                    print(name, end="\r")
                    inverse_transformation_matrix = np.linalg.inv(np.array(dico_infos[name]["transformation_matrix"], dtype=np.float32))
                    patch_shape = patch.shape[-2:]
                    patch: np.ndarray = np.reshape(patch,(1,*patch.shape))
                    with torch.no_grad():
                        model.eval()
                        prediction: np.ndarray = transfert_from_pytorch(
                            model(transfert_to_pytorch(
                                np.array(self.standardizer.standardize(patch),dtype=np.float32)
                            ))
                        )
                        del patch

                        prediction = prediction.flatten()
                        if len(prediction) > 3:
                            raise Exception("Cannot show rgb overlay with more than 3 classes")
                        # compute manually points to slice
                        corner_upleft = np.array([0,0,1])
                        mapped_corner_upperleft = np.round(inverse_transformation_matrix.dot(corner_upleft)).astype(np.int32).tolist()
                        corner_bottomright = np.array([*patch_shape[::-1],1])
                        mapped_corner_bottomright = np.round(inverse_transformation_matrix.dot(corner_bottomright)).astype(np.int32).tolist()

                        for i in range(len(prediction)):
                            overlay_pred[mapped_corner_upperleft[1]:mapped_corner_bottomright[1],
                                         mapped_corner_upperleft[0]:mapped_corner_bottomright[0],i] *= 1 if prediction[i] > 0.5 else 0

                overlay_pred = reconstructed_image + self.normalize(overlay_pred) * (1 - threshold)
                name = "pred"
                plt.figure(2)
                plt.title(f"Overlay " + name)
                plt.imshow(np.array(overlay_pred*255,dtype=np.uint8))
                plt.savefig(FolderInfos.base_filename + f"rgb_overlay_{name}.png")
                del overlay_pred
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


    def __call__(self, model,device):
        model.load_state_dict(torch.load(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_out\2021-07-21_02h37min54s_\2021-07-21_02h37min54s__model_epoch-14_it-15923.pt"))
        self.generate_overlay_matrices(self.name_img,model=model,device=device,threshold=0.5)
