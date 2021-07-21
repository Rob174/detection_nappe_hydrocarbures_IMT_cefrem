import json

import cv2
import numpy as np
from h5py import File
import matplotlib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\images_preprocessed.hdf5","r") as cache_source:
        img_source = np.array(cache_source["027481_0319CB_0EB7"],dtype=np.float32)
        min = np.min(img_source)
        max = np.max(img_source)
        plt.figure(1)
        plt.title(f"Source image")
        plt.imshow(img_source,cmap="gray")
        # plt.savefig(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_test\outputs\source_img.png")

    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_cache_images1.hdf5","r") as cache_images:
        with open(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_img_infos1.json","r") as fp:
            infos = json.load(fp)
        for i,[key,img] in enumerate(cache_images.items()):
            transfo = np.array(infos[key]["transformation_matrix"],dtype=np.float32)
            transfo_inverse = np.linalg.inv(transfo)
            pt = np.array([0,0,1],dtype=np.float32)
            pt_mapped = transfo_inverse.dot(pt)
            pt_mapped = np.round(pt_mapped)
            col, row,_ = pt_mapped
            col = int(col)-4
            row = int(row)
            img = np.array(img,dtype=np.float32)[0,:,:]
            src = cv2.resize(img_source[row:row+1000,col:col+1000],dsize=(256,256),interpolation=cv2.INTER_LANCZOS4)
            diff = img-src
            print(f"Diff abs max {np.max(np.abs(diff))}")
            plt.figure(i*2+2)
            plt.title(f"Cache {key} loc: {pt_mapped.tolist()}")
            plt.imshow(img,cmap="gray",vmin=min,vmax=max)
            plt.figure(i*2+1+2)
            plt.title(f"Cache {key} loc: {pt_mapped.tolist()}")
            plt.imshow(diff,cmap="gray",vmin=min,vmax=max)
            # plt.savefig(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_test\outputs\patch"+key+".png")
        plt.show()
        b=0





