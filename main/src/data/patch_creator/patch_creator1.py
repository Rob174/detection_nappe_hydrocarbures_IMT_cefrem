from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np
import cv2


class Patch_creator1(Patch_creator0):
    def __init__(self, grid_size_px, images_informations_preprocessed, test=False):
        super(Patch_creator1, self).__init__(grid_size_px, images_informations_preprocessed, test)
    def transform_back(self,image: np.ndarray, name: str) -> np.ndarray:
        # doc https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
        transformation_matrix = np.array(self.images_informations_preprocessed[name]["transform"])
        array = cv2.warpAffine(image,transformation_matrix,(*image.shape,)[::-1])
        return np.array(array)

    def __call__(self, image: np.ndarray,image_name: str, patch_id: int,count_reso=False) -> np.ndarray:
        super(Patch_creator1, self).__call__(self.transform_back(image),image_name,patch_id,count_reso)
