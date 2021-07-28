import json

import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    path_images = FolderInfos.input_data_folder + "filtered_cache_images.hdf5"
    path_annotations = FolderInfos.input_data_folder + "filtered_cache_annotations.hdf5"
    path_infos = FolderInfos.input_data_folder + "filtered_img_infos.json"
    new_path_images = FolderInfos.input_data_folder + "_filtered_cache_images.hdf5"
    new_path_annotations = FolderInfos.input_data_folder + "_filtered_cache_annotations.hdf5"
    exclude_keys = []
    with File(path_annotations, "r") as annotations:
        keys = list(annotations.keys())
        threshold = 10
        for name, annotation in annotations.items():
            annotation = np.array(annotation)
            non_zeros = np.count_nonzero(annotation)
            if non_zeros <= threshold:
                exclude_keys.append(name)

    print(len(exclude_keys))
    with File(path_images, "r") as images_src:
        with File(new_path_images, "w") as images_dest:
            for k in images_src:
                if k not in exclude_keys:
                    array = np.array(images_src[k], dtype=np.float32)
                    images_dest.create_dataset(k, shape=array.shape, dtype='f', data=array)
            images_dest.flush()
    with File(path_annotations, "r") as annotations_src:
        with File(new_path_annotations, "w") as annotations_dest:
            for k in annotations_src:
                if k not in exclude_keys:
                    array = np.array(annotations_src[k], dtype=np.uint8)
                    annotations_dest.create_dataset(k, shape=array.shape, dtype='i', data=array)
            annotations_dest.flush()
    with open(path_infos, "r") as fp:
        infos = json.load(fp)
    new_dico = {}
    with open(path_infos, "w") as fp:
        for k in keys:
            new_dico[k] = infos[k]
        json.dump(new_dico, fp)
