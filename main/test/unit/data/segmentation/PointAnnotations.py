from h5py import File
import numpy as np
import unittest
import matplotlib.pyplot as plt

from main.FolderInfos import FolderInfos
from main.src.data.Annotations.PointAnnotations import PointAnnotations

def plot_differences(array1, array2, cmap1="gray", cmap2="gray"):
    fig1 = plt.figure(1)
    fig1.canvas.manager.window.move(0, 0)
    plt.imshow(array1,cmap=cmap1)
    fig2 = plt.figure(2)
    fig2.canvas.manager.window.move(1000, 0)
    plt.imshow(array2,cmap=cmap2)
    plt.show()

class TestPointAnnotations(unittest.TestCase):
    def __init__(self,*args,**kwargs):
        super(TestPointAnnotations, self).__init__(*args,**kwargs)
        self.img_test = "027481_0319CB_0EB7"
    def test_get_identity(self):
        FolderInfos.init(test_without_data=True)
        with File(FolderInfos.input_data_folder+"annotations_labels_preprocessed.hdf5","r") as annotation:
            originale = np.array(annotation[self.img_test],dtype=np.float32)
        point_annotations = PointAnnotations()
        side_length = originale.shape[0]
        annotation = point_annotations.get(self.img_test,np.identity(3),side_length)
        originale = originale[:side_length,:side_length]

        self.assertGreater(np.count_nonzero(annotation),0,msg="Annotation contains only 0. Nothing has been drawn")

        try:
            np.testing.assert_array_equal(originale,annotation)
        except AssertionError:
            plot_differences(originale, annotation)
            raise AssertionError
if __name__ == "__main__":
    unittest.main()