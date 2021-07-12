import json

import h5py

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass


class Trainer0(BaseClass):
    """Class managing the training process

    Args:
        dataset: DatasetFactory object to access data
        saver: Saver0 object (see its documentation
    """

    def __init__(self,
                 dataset,
                 saver,
                 *args, **kwargs):

        self.dataset = dataset
        self.saver = saver

        self.attr_last_iter = -1
        self.attr_last_epoch = -1
        self.attr_global_name = "trainer"
        self.saver(self).save()

    def __call__(self):
        return self.call()

    def call(self):
        """Save the input and outputs with metadata"""
        with h5py.File(FolderInfos.input_data_folder + "filtered_other_cache_images.hdf5", "w") as cache_images:
            dico_info = {}
            cache_filtered_cache_annotation_num_patches = 16519
            class IterWrapper:
                def __init__(self,dataset_factory,dataset):
                    self.dataset = dataset
                    self.dataset_factory = dataset_factory
                def __iter__(self):
                    return self.dataset_factory.__iter__(self.dataset)
            data = IterWrapper(self.dataset,"tr")
            for i, [input, output, transformation_matrix, item] in enumerate(data):
                dico_info[str(i)] = {"source_img": item, "transformation_matrix": transformation_matrix.tolist()}
                cache_images.create_dataset(str(i), shape=input.shape, dtype='f', data=input)
                if i % 100 == 0:
                    print(f"i={i}")
                    with open(FolderInfos.input_data_folder + "filtered_other_img_infos.json", "w") as fp:
                        json.dump(dico_info, fp)
                if i > 16519:
                    break

            with open(FolderInfos.input_data_folder + "filtered_img_infos.json", "w") as fp:
                json.dump(dico_info, fp)

        self.saver(self.dataset)
        self.saver(self).save()