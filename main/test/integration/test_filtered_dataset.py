import json
import unittest
import random

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
    def test_same_keys(self):
        with File(self.path_images,"r") as cache:
            keys_images = set(cache.keys())
        with File(self.path_annotations,"r") as cache:
            keys_annot = set(cache.keys())
        with open(self.path_infos,"r") as fp:
            keys_infos = set(json.load(fp).keys())
        self.assertTrue(keys_images==keys_annot,msg=f"images contains \n{keys_images.difference(keys_annot)} \nnot in annotations" if len(keys_images) > len(keys_annot) else f"annotations contains \n{keys_annot.difference(keys_images)} \nnot in images")
        self.assertTrue(keys_images==keys_infos,msg=f"images contains \n{keys_images.difference(keys_infos)} \nnot in infos" if len(keys_images) > len(keys_infos) else f"infos contains \n{keys_infos.difference(keys_images)} \nnot in images")
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
            nb = 0
            for name,annotation in annotations.items():
                annotation = np.array(annotation)
                non_zeros = np.count_nonzero(annotation)
                try:
                    self.assertTrue(non_zeros > threshold)
                except AssertionError:
                    nb += 1
                    # plot_differences(images[name],annotation)
            if nb > 0:
                raise AssertionError(f"{nb} over {len(annotations)} images have no annotations")
    def test_visual(self):
        nb_random_keys = 5
        with File(self.path_annotations,"r") as annotations:
            with File(self.path_images,"r") as images:
                print(f"{len(images)} images and {len(annotations)} annotations")
                keys = list(images.keys())
                random.shuffle(keys)
                keys = keys[:nb_random_keys]
                for key in keys:
                    image = images[key]
                    annotation = annotations[key]
                    # plot_differences(image,annotation)
                    # self.assertTrue(input("OK label? ") != "f")



