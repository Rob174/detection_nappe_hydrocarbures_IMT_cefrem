import unittest

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache


class MyTestCase(unittest.TestCase):
    def test_compile(self):
        FolderInfos.init(test_without_data=True)
        images, annotations, informations = FabricFilteredCache()()


if __name__ == '__main__':
    unittest.main()
