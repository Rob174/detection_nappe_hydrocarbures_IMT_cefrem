import numpy as np
import unittest
import random

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricFilteredCacheOther import FabricFilteredCacheOther
from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.MarginCheck import MarginCheck
from main.src.data.balance_classes.BalanceClassesNoOther import BalanceClassesNoOther
from main.src.data.balance_classes.BalanceClassesOnlyOther import BalanceClassesOnlyOther


class MyTestCase(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        FolderInfos.init(test_without_data=True)
        self.num_random_test = 50
        self.image_size = 256

    def test_compile(self):
        images, annotations, informations = FabricFilteredCacheOther()()
    def test_shape_images(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        ids = images.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(images.get(id).shape,(self.image_size,self.image_size),"images do not have the same shape")
    def test_shape_matrix(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        ids = images.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(np.array(informations[id]["transformation_matrix"]).shape,(3,3))
    def test_shape_annotations(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        ids = annotations.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(annotations.get(id).shape,(self.image_size,self.image_size),"annotation do not have the same shape")
    def test_same_ids(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        k_img = set(images.keys())
        k_annot = set(annotations.keys())
        k_json = set(informations.keys())
        self.assertEqual(k_img,k_annot,"not the same keys in images and annotations")
        self.assertEqual(k_img,k_json,"not the same keys in images and informations")
    def test_not_in_margins(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        ids = images.keys()
        random.shuffle(ids)
        balance = MarginCheck(threshold=10)
        for id in ids[:self.num_random_test]:
            image = images.get(id)
            self.assertEqual(balance.check_reject(image),False, "in margin problem")
    def test_all_other(self):
        images, annotations, informations = FabricFilteredCacheOther()()
        ids = images.keys()
        random.shuffle(ids)
        balance = BalanceClassesOnlyOther(other_index=0)
        label_modifier = LabelModifier0(class_mapping=annotations.attr_mapping)
        for id in ids[:self.num_random_test]:
            annotation = annotations.get(id)
            annotation = label_modifier.make_classification_label(annotation)
            self.assertEqual(balance.filter(annotation),False,"balance issue")







if __name__ == '__main__':
    unittest.main()
