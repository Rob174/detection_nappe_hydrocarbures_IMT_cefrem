"""Class dedicated to create a dataset of patches. Open for modifications to generate custom cache"""

import json

import h5py

from main.FolderInfos import FolderInfos as FI
from main.src.param_savers.BaseClass import BaseClass


class TrainerGenerateCache(BaseClass):
    """Class dedicated to create a dataset of patches. Open for modifications to generate custom cache

    Args:
        dataset: DatasetFactory object to access data
        saver: Saver0 object (see its documentation
    """

    def __init__(self,
                 dataset,
                 *args, **kwargs):

        self.dataset = dataset

        self.attr_last_iter = -1
        self.attr_last_epoch = -1
        self.attr_global_name = "trainer"

    def __call__(self, name: str):
        return self.call(name)

    def call(self, name: str):
        """Save the input and outputs with metadata

        Args:
            name: str, identifier of the cache

        Returns:

        """
        print(self.__class__.__name__ + " chosen. Launching loop...")
        with h5py.File(FI.input_data_folder + name + "_image.hdf5", "w") as cache_images:
            with h5py.File(FI.input_data_folder + name + "_annotations.hdf5", "w") as cache_annotations:
                dico_info = {}

                class Wrapper:
                    def __init__(self, data, dataset):
                        self.data = data
                        self.dataset = dataset

                    def __iter__(self):
                        return self.data.__iter__(self.dataset)

                for i, [input, output, transformation_matrix, item] in enumerate(Wrapper(self.dataset, "all")):
                    dico_info[str(i)] = {"source_img": item, "transformation_matrix": transformation_matrix.tolist()}
                    cache_images.create_dataset(str(i), shape=input.shape, dtype='f', data=input)
                    cache_annotations.create_dataset(str(i), shape=output.shape, dtype='i', data=output)
                    if i % 1000 == 0:
                        print(i, end="\r")
                        with open(FI.input_data_folder + name + "_img_infos.json", "w") as fp:
                            json.dump(dico_info, fp)
                self.dataset.attr_dataset.attr_image_dataset.close()

        with open(FI.input_data_folder + name + "_img_infos.json", "w") as fp:
            json.dump(dico_info, fp)

