"""Class dedicated to create a dataset of patches. Open for modifications to generate custom cache"""


import json

import h5py
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass


class Trainer0(BaseClass):
    """Class dedicated to create a dataset of patches. Open for modifications to generate custom cache

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
        with h5py.File(FolderInfos.input_data_folder + "test_cache_image.hdf5", "w") as cache_images:
            with h5py.File(FolderInfos.input_data_folder + "test_cache_annotations.hdf5", "w") as cache_annotations:
                dico_info = {}

                class Wrapper:
                    def __init__(self,data,dataset):
                        self.data = data
                        self.dataset = dataset
                    def __iter__(self):
                        return self.data.__iter__(self.dataset)

                for i, [input, output, transformation_matrix, item] in enumerate(Wrapper(self.dataset,["027481_0319CB_0EB7"])):
                    dico_info[str(i)] = {"source_img": item, "transformation_matrix": transformation_matrix.tolist()}
                    cache_images.create_dataset(str(i), shape=input.shape, dtype='f', data=input)
                    cache_annotations.create_dataset(str(i), shape=output.shape, dtype='i', data=output)
                    if i % 1000 == 0:
                        print(i,end="\r")
                        with open(FolderInfos.input_data_folder + "test_cache_img_infos.json", "w") as fp:
                            json.dump(dico_info, fp)

        with open(FolderInfos.input_data_folder + "test_cache_img_infos.json", "w") as fp:
            json.dump(dico_info, fp)

        self.saver(self.dataset)
        self.saver(self).save()
