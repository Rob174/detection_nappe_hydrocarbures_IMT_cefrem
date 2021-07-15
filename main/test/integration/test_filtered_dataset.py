import json
import unittest
import cv2

import numpy as np

from h5py import File

from main.FolderInfos import FolderInfos
from main.test.unit.data.segmentation.PointAnnotations import plot_differences


class TestCache(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestCache, self).__init__(*args,**kwargs)
        FolderInfos.init(test_without_data=True)
        self.path_images = FolderInfos.input_data_folder+"filtered_cache_images.hdf5"
        self.path_annotations = FolderInfos.input_data_folder+"filtered_cache_annotations.hdf5"
        self.path_infos = FolderInfos.input_data_folder+"filtered_img_infos.json"
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
        with File(self.path_annotations,"r") as annotations:
            for annotation in annotations.values():
                annotation = np.array(annotation)
                non_zeros = np.count_nonzero(annotation)
                self.assertTrue(non_zeros > threshold)
    def test_visual(self):
        with File(self.path_annotations,"r") as annotations:
            with File(self.path_images,"r") as images:
                print(f"{len(images)} images and {len(annotations)} annotations")
                for image,annotation in zip(images.values(),annotations.values()):
                    plot_differences(image,annotation)
                    self.assertTrue(input("OK label? ") != "f")
    def test_confusion_matrix(self):
        out_size = 10000
        with File(self.path_images,"r") as images:
            with open(self.path_infos,"r") as fp:
                infos = json.load(fp)
            for image,matrix in zip(images.values(),map(lambda x:np.array(x["transformation_matrix"],dtype=np.float32),infos.values())):
                image = np.array(image,dtype=np.float32)
                plot_differences(image,cv2.warpAffine(image,matrix[:-1,:],flags=cv2.WARP_INVERSE_MAP,dsize=(out_size,out_size)))
                self.assertTrue(input("OK transformed_back? ") != "f")



