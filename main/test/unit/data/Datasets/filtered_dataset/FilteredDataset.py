import numpy as np
import unittest
import random

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache
from main.src.data.MarginCheck import MarginCheck
from main.src.data.balance_classes.BalanceClassesNoOther import BalanceClassesNoOther


class MyTestCase(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        FolderInfos.init(test_without_data=True)
        self.num_random_test = 50
        self.image_size = 256

    def test_compile(self):
        images, annotations, informations = FabricFilteredCache()()
    def test_shape_images(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = images.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(images.get(id).shape,(self.image_size,self.image_size))
    def test_shape_matrix(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = images.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(np.array(informations[id]["transformation_matrix"]).shape,(3,3))

    def test_shape_annotations(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = annotations.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(annotations.get(id).shape,(self.image_size,self.image_size))
    def test_same_ids(self):
        images, annotations, informations = FabricFilteredCache()()
        k_img = set(images.keys())
        k_annot = set(annotations.keys())
        k_json = set(informations.keys())
        self.assertEqual(k_img,k_annot)
        self.assertEqual(k_img,k_json)
    def test_not_in_margins(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = images.keys()
        random.shuffle(ids)
        balance = MarginCheck(threshold=10)
        for id in ids[:self.num_random_test]:
            image = images.get(id)
            self.assertEqual(balance.check_reject(image),False)
    def test_no_other(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = images.keys()
        random.shuffle(ids)
        balance = BalanceClassesNoOther(other_index=0)
        for id in ids[:self.num_random_test]:
            image = images.get(id)
            self.assertEqual(balance.filter(image),False)







if __name__ == '__main__':
    unittest.main()
