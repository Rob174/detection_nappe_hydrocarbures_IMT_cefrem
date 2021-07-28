import unittest
import random

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache


class MyTestCase(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        FolderInfos.init(test_without_data=True)
        self.num_random_test = 50

    def test_compile(self):
        images, annotations, informations = FabricFilteredCache()()
    def test_shape_images(self):
        images, annotations, informations = FabricFilteredCache()()
        ids = images.keys()
        random.shuffle(ids)
        for id in ids[:self.num_random_test]:
            self.assertEqual(images.get(id).shape,(256,256))



if __name__ == '__main__':
    unittest.main()
