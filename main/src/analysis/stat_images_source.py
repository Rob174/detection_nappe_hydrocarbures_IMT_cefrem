"""Goal : compare mean and std of all images of the source dataset"""
import numpy as np
from h5py import File

if __name__ == '__main__':

    with File(
            r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\images_preprocessed.hdf5",
            "r") as cache_img:

        stats = np.zeros((len(cache_img), 2),
                         dtype=np.float32)  # containing at the first index mean and std at the second one
        dico_corresp = {}
        for i, [name_id, image] in enumerate(cache_img.items()):
            image = np.array(image, dtype=np.float32)
            dico_corresp[name_id] = i
            stats[i] = np.array([np.mean(image), np.std(image)])
            if i % 100 == 0:
                print(f"i={i}", end="\r")
        b = 0
