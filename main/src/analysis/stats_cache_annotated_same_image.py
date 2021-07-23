import json

from h5py import File

import numpy as np

if __name__ == '__main__':
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_other_cache_images.hdf5","r") as cache_patch:
        with open(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_other_img_infos.json","r") as fp:
            dico_infos = json.load(fp)
        buffer = []
        last_src_image = ""
        num_original_images = len(list({v for v in list(map(lambda x:dico_infos[x]["source_img"],dico_infos.keys()))}))
        stats = []  # containing at the first index mean and std at the second one
        dico_corresp = {}
        id_group = 0
        for name in dico_infos.keys():
            patch = cache_patch[name]
            patch = np.array(patch,dtype=np.float32)
            source_image = dico_infos[name]["source_img"] # Patch extracted from the same augmented image as the previous one

            if source_image != last_src_image and last_src_image != "": # End of patches with the same augmentations
                buffer_npy = np.stack(buffer,axis=0)
                stats.append([np.mean(buffer_npy),np.std(buffer_npy)])
                if source_image not in dico_corresp:
                    dico_corresp[source_image] = []
                dico_corresp[source_image].append(id_group)
                last_src_image = source_image
                buffer = [patch]
                id_group += 1
            else:
                buffer.append(patch)
                last_src_image = source_image
        b=0
