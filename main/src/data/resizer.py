import cv2
import numpy as np
import matplotlib.pyplot as plt

from main.FolderInfos import FolderInfos
import main.src.data.segmentation.DataSentinel1Segmentation as data
from PIL import Image


class Resizer:
    def __init__(self,out_size_w=None,interpolation=None):
        self.attr_out_size_w = out_size_w
        self.attr_interpolation = interpolation
    def __call__(self,array) -> np.ndarray:
        if self.attr_out_size_w is None:
            return array
        img = Image.fromarray(((array-np.min(array))/(np.max(array)-np.min(array))*255).astype(np.uint8))
        # resize: warning: .resize((width !!,height !!)) in this order
        img = img.resize((self.attr_out_size_w, int(self.attr_out_size_w / array.shape[1] * array.shape[0])),
                             resample=self.attr_interpolation)
        return np.array(img,dtype=np.float32)

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    folder = FolderInfos.root_folder + "test_out" + FolderInfos.separator + "resizer" + FolderInfos.separator

    dataset = data.DataSentinel1Segmentation()
    img,label = dataset[0]
    plt.figure()
    plt.title("Original image")
    plt.imshow(img,cmap="gray")
    plt.savefig(folder+f"{dataset.current_name}_original.png")
    plt.clf()
    plt.figure()
    plt.title("Resized 256 px image with nearest neigbours interpolation")
    plt.imshow(Resizer(out_size_w=256,interpolation=Image.NEAREST)(img),cmap="gray")
    plt.savefig(folder + f"{dataset.current_name}_resized_nearest_pillow.png")