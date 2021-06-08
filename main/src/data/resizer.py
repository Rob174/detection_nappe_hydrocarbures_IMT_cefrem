import cv2
import numpy as np
import matplotlib.pyplot as plt

from main.FolderInfos import FolderInfos
import main.src.data.segmentation.DataSentinel1Segmentation as data
from main.src.param_savers.BaseClass import BaseClass


class Resizer(BaseClass):
    def __init__(self,out_size_w,interpolation=None):
        self.attr_out_size_w = out_size_w
        self.attr_interpolation = interpolation
    def __call__(self,array) -> np.ndarray:
        if self.attr_out_size_w == array.shape[1]:
            return array
        # resize: warning: cv2.resize(array,(width !!,height !!)) in this order
        array = np.array(array,dtype=np.float64)
        array = cv2.resize(src=array, dsize=(self.attr_out_size_w, int(self.attr_out_size_w / array.shape[1] * array.shape[0])),
                           interpolation=self.attr_interpolation)
        return np.array(array,dtype=np.float32)

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    folder = FolderInfos.root_folder + "test_out" + FolderInfos.separator + "resizer" + FolderInfos.separator

    dataset = data.DataSentinel1Segmentation()
    img,label = dataset[0]
    plt.figure()
    plt.title("Original image")
    plt.imshow(img,cmap="gray")
    plt.savefig(folder+f"{dataset.current_name}_original.png")
    plt.figure()
    plt.title("Resized 256 px image with no interpolation")
    plt.imshow(Resizer(out_size_w=256)(img),cmap="gray",vmin=np.min(img),vmax=np.max(img))
    plt.savefig(folder + f"{dataset.current_name}_resized_no_interp.png")
    plt.clf()
    plt.figure()
    plt.title("Resized 256 px image with nearest neigbours interpolation")
    plt.imshow(Resizer(out_size_w=256,interpolation=cv2.INTER_NEAREST)(img),cmap="gray",vmin=np.min(img),vmax=np.max(img))
    plt.savefig(folder + f"{dataset.current_name}_resized_nearest.png")
    plt.clf()
    plt.figure()
    plt.title("Resized 256 px image with LANCZOS4 interpolation")
    plt.imshow(Resizer(out_size_w=256,interpolation=cv2.INTER_LANCZOS4)(img),cmap="gray",vmin=np.min(img),vmax=np.max(img))
    plt.savefig(folder + f"{dataset.current_name}_resized_lanczos4.png")
    plt.clf()
    plt.figure()
    plt.title("Resized 256 px image with LINEAR interpolation")
    plt.imshow(Resizer(out_size_w=256,interpolation=cv2.INTER_LINEAR)(img),cmap="gray",vmin=np.min(img),vmax=np.max(img))
    plt.savefig(folder + f"{dataset.current_name}_resized_linear.png")
    plt.clf()
    plt.figure()
    plt.title("Resized 256 px image with CUBIC interpolation")
    plt.imshow(Resizer(out_size_w=256,interpolation=cv2.INTER_CUBIC)(img),cmap="gray",vmin=np.min(img),vmax=np.max(img))
    plt.savefig(folder + f"{dataset.current_name}_resized_cubic.png")
    plt.clf()