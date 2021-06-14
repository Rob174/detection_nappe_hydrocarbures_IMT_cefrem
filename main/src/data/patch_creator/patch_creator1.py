from main.FolderInfos import FolderInfos
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np
import cv2
import functools



class Patch_creator1(Patch_creator0):
    def __init__(self, grid_size_px, images_informations_preprocessed, test=False):
        super(Patch_creator1, self).__init__(grid_size_px, images_informations_preprocessed, test)

    # @functools.lru_cache(maxsize=1)
    def transform_back(self, image: np.ndarray, name: str) -> np.ndarray:
        # doc https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
        transformation_matrix = np.array(self.images_informations_preprocessed[name]["transform"],dtype=np.float32)
        extreme_point = (np.array([*image.shape[:2],2])-1)
        extreme_point_transform = transformation_matrix.dot(extreme_point)
        factorx = image.shape[0]/extreme_point_transform[0]
        factory = image.shape[1]/extreme_point_transform[1]
        biggest_factor = min(factorx,factory)
        scaling_up = np.array([[biggest_factor,0,0],[0,biggest_factor,0],[0,0,1]])
        transformation_matrix = scaling_up.dot(transformation_matrix)[:-1,:]
        array = cv2.warpAffine(image, transformation_matrix,dsize=image.shape)
        return array

    def __call__(self, image: np.ndarray, image_name: str, patch_id: int, count_reso: bool = False) -> np.ndarray:
        self.last_image = self.transform_back(image,image_name)
        return super(Patch_creator1, self).__call__(self.last_image, image_name, patch_id, count_reso)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    FolderInfos.init(test_without_data=True)

    import matplotlib.pyplot as plt
    import os
    from main.src.data.DatasetFactory import DatasetFactory

    folder = FolderInfos.data_test+"outputs"+FolderInfos.separator+"Patch_creator1"+FolderInfos.separator
    if os.path.exists(folder) is False:
        os.mkdir(folder)


    from PIL import Image, ImageDraw
    for grid_size in [500,1000,1500]:
        dataset_factory = DatasetFactory(dataset_name="sentinel1",
                                                           usage_type="classification",
                                                           patch_creator="straight_fixed_px",
                                                           grid_size=grid_size,
                                                           input_size=256)  # get the first test image
        name_img = "027481_0319CB_0EB7"
        patches = dataset_factory.attr_dataset.make_patches_of_image(name_img)
        del patches
        array = dataset_factory.attr_patch_creator.last_image
        plt.figure()  # Create new separated figure
        plt.imshow(array-np.min(array), cmap="gray",vmin=0,vmax=np.max(array))  #
        plt.savefig(folder + f"{name_img}_original.png")
        dico_infos = dataset_factory.attr_dataset.images_infos[name_img]
        plt.clf()
        plt.figure() # Create new separated figure
        plt.imshow(array-np.min(array),cmap="gray",vmin=0,vmax=np.max(array)) #
        plt.savefig(folder+f"{name_img}_original_transformed_{grid_size}.png")

        plt.clf() # Clear previous figures
        plt.figure() # Create new separated figure
        # Convert the raster array to a 0-255 array
        image_rgb_uint8 = (array - np.min(array)) / (np.max(array) - np.min(array))*255 # Normalisation
        image_rgb_uint8 = image_rgb_uint8.astype(np.uint8)# Convert to uint8 array
        image_rgb_uint8 = np.stack((image_rgb_uint8,)*3,axis=-1)
        image_cpy = Image.fromarray(image_rgb_uint8) # Convert to pillow object
        image_cpy1 = Image.fromarray(np.copy(image_rgb_uint8)) # Prepare the image to draw on
        draw = ImageDraw.ImageDraw(image_cpy)
        for coords in dataset_factory.attr_patch_creator.coords:
            draw.rectangle(coords, width=30, outline="red")  # Draw the patches
        plt.clf()  # Clear previous figures
        plt.figure()  # Create new separated figure
        image_annotated = Image.blend(image_cpy, image_cpy1, 0.5)  # Mix the original image with the annotated one
        plt.imshow(image_annotated)  # Show it
        plt.title(f"Patches of {grid_size} px length on {name_img}")
        plt.savefig(
            folder + f"{name_img}_with_patches_size-{grid_size}.png")  # Save the current figure to file