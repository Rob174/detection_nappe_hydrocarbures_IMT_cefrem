import json

from main.FolderInfos import FolderInfos
from h5py import File
import numpy as np
import matplotlib.pyplot as plt
import cv2
if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    path_raster_cache_orig = FolderInfos.input_data_folder+"images_preprocessed.hdf5"
    path_raster_cache_filtered = FolderInfos.input_data_folder+"filtered_cache_images.hdf5"
    path_annotations_cache_orig = FolderInfos.input_data_folder+"annotations_labels_preprocessed.hdf5"
    path_annotations_cache_filtered = FolderInfos.input_data_folder+"filtered_cache_annotations.hdf5"

    path_image_infos_filtered = FolderInfos.input_data_folder+"filtered_img_infos.json"
    key_source_img_ref = "027481_0319CB_0EB7"
    key_patch = "1026"



    with open(path_image_infos_filtered,"r") as fp:
        dico_infos = json.load(fp)
    transformation_matrix = np.array(dico_infos[key_patch]["transformation_matrix"],dtype=np.float32)[:-1,:]

    with File(path_raster_cache_orig,"r") as cache:
        image = np.array(cache[key_source_img_ref],dtype=np.float32)
        plt.figure(1)
        plt.title("Image from original cache")
        plt.imshow(image,cmap="gray")
        image = cv2.warpAffine(image,transformation_matrix,dsize=(256,256),flags=cv2.INTER_LANCZOS4)
        plt.figure(2)
        plt.title("Image from original cache with warpAffine")
        plt.imshow(image,cmap="gray")

    with File(path_raster_cache_filtered,"r") as cache:
        image = np.array(cache[key_patch],dtype=np.float32)
        plt.figure(3)
        plt.title("Image from cache filtered")
        plt.imshow(image,cmap="gray")

    plt.show()

