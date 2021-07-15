import unittest

import numpy as np

from h5py import File

from main.FolderInfos import FolderInfos


class TestCache(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestCache, self).__init__(*args,**kwargs)
        FolderInfos.init(test_without_data=True)
        self.path_images = FolderInfos.input_data_folder+"filtered_cache_images.hdf5"
        self.path_annotations = FolderInfos.input_data_folder+"filtered_cache_annotations.hdf5"
        self.margin_threshold = 100
        self.size = 256
    def test_shape(self):
        with File(self.path_images,"r") as images:
            for image in images.values():
                self.assertEqual(image.shape,(self.size,self.size))
        with File(self.path_annotations,"r") as annotations:
            for annotation in annotations.values():
                self.assertEqual(annotation.shape,(self.size,self.size))

    def test_no_margins(self):
        threshold = 100
        with File(self.path_images,"r") as images:
            for image in images.values():
                image = np.array(image)
                self.assertTrue(np.count_nonzero(image[image == 0]) <= threshold)
    def test_contains_annotations(self):
        threshold = 10
        with File(self.path_images,"r") as annotations:
            for annotation in annotations.values():
                annotation = np.array(annotation)
                non_zeros = np.count_nonzero(annotation)
                print(non_zeros)
                self.assertTrue(non_zeros > threshold)
