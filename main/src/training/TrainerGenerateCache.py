import json

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass
import h5py


class TrainerGenerateCache(BaseClass):
    """Class managing the training process

    Args:
        dataset: DatasetFactory object to access data
        saver: Saver0 object (see its documentation
    """
    def __init__(self ,
                 dataset,
                 saver,
                 *args,**kwargs):

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
        self.saver(self.dataset)
        self.saver(self).save()
        with h5py.File(FolderInfos.input_data_folder+"filtered_cache_images.hdf5","w") as cache_images:
            with h5py.File(FolderInfos.input_data_folder+"filtered_cache_annotations.hdf5","w") as cache_annotations:
                dico_info = {}
                for i, [input, output,transformation_matrix,item] in enumerate(self.dataset.__iter__(dataset="tr")):
                    dico_info[str(i)] = {"source_img":item, "transformation_matrix":transformation_matrix.tolist()}
                    cache_images.create_dataset(str(i),shape=input.shape ,dtype='f' ,data=input)
                    cache_annotations.create_dataset(str(i),shape=output.shape ,dtype='i' ,data=output)
                    print("\rsave %d" % i)
                    if i % 100 == 0:
                        print("\rsave step %d" % i)
                        with open(FolderInfos.input_data_folder+"filtered_img_infos.json","w") as fp:
                            json.dump(dico_info,fp,indent=4)

                with open(FolderInfos.input_data_folder+"filtered_img_infos.json","w") as fp:
                    json.dump(dico_info, fp,indent=4)

        self.saver(self.dataset)
        self.saver(self).save()