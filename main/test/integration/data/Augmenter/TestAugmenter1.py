import unittest
import numpy as np
from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
from main.src.data.segmentation.PointAnnotations import PointAnnotations


class TestAugmenter1(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestAugmenter1, self).__init__(*args,**kwargs)
        self.img_test = "027481_0319CB_0EB7"
        FolderInfos.init(test_without_data=True)
        with File(FolderInfos.input_data_folder+"annotations_labels_preprocessed.hdf5","r") as annotation:
            self.originale = np.array(annotation[self.img_test],dtype=np.float32)
    def test_grid(self):

        augmenter = Augmenter1(1000,256,"combinedRotResizeMir_10_0.25_4")
        grid = augmenter.get_grid(self.originale.shape,np.identity(3))
        grid_rows_expected = [0.0, 256.0, 512.0, 768.0, 1024.0, 1280.0, 1536.0, 1792.0, 2048.0, 2304.0, 2560.0, 2816.0,
                           3072.0, 3328.0, 3584.0, 3840.0, 4096.0, 4352.0, 4608.0, 4864.0, 5120.0, 5376.0, 5632.0, 5888.0,
                           6144.0, 6400.0, 6656.0, 6912.0, 7168.0, 7424.0, 7680.0, 7936.0, 8192.0, 8448.0, 8704.0, 8960.0,
                           9216.0, 9472.0, 9728.0, 9984.0, 10240.0]
        grid_cols_expected = [0.0, 256.0, 512.0, 768.0, 1024.0, 1280.0, 1536.0, 1792.0, 2048.0, 2304.0, 2560.0, 2816.0,
                              3072.0, 3328.0, 3584.0, 3840.0, 4096.0, 4352.0, 4608.0, 4864.0, 5120.0, 5376.0, 5632.0,
                              5888.0, 6144.0, 6400.0, 6656.0, 6912.0, 7168.0, 7424.0, 7680.0, 7936.0, 8192.0, 8448.0,
                              8704.0, 8960.0, 9216.0, 9472.0, 9728.0, 9984.0, 10240.0, 10496.0, 10752.0, 11008.0, 11264.0,
                              11520.0, 11776.0, 12032.0, 12288.0, 12544.0, 12800.0, 13056.0, 13312.0, 13568.0, 13824.0, 14080.0,
                              14336.0, 14592.0, 14848.0, 15104.0, 15360.0, 15616.0, 15872.0, 16128.0, 16384.0, 16640.0, 16896.0,
                              17152.0, 17408.0, 17664.0, 17920.0, 18176.0]
        coords = list(zip(*list(x.flat for x in np.meshgrid(grid_rows_expected, grid_cols_expected))))
        np.testing.assert_array_equal(grid,coords)
    def test_patch_annotation_with_cache(self):
        augmenter = Augmenter1(1000,256,"combinedRotResizeMir_10_0.25_4")
        # Zone 1: row and cols index start of patch in test_grid lists
        coords = [(5632,4352),(5632,4608),(5888,4352),(5888,4608)]
        for coord in coords:
            self.assertTrue(np.count_nonzero(augmenter.transform_image(self.originale,np.identity(3),coord)[0]) > 0)
    def test_patch_annotation_points(self):
        augmenter = Augmenter1(1000,256,"combinedRotResizeMir_10_0.25_4")
        annotations = PointAnnotations()
        # Zone 1: row and cols index start of patch in test_grid lists
        coords = [(5632,4352),(5632,4608),(5888,4352),(5888,4608)]
        for coord in coords:
            patch,_ = augmenter.transform_label(annotations.get,self.img_test,np.identity(3),coord)
            self.assertTrue(np.count_nonzero(patch) > 0)
            np.testing.assert_array_equal(augmenter.transform_image(self.originale,np.identity(3),coord)[0],
                                          patch)
A



if __name__ == "__main__":
    unittest.main()