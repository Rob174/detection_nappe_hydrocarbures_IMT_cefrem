from main.FolderInfos import FolderInfos
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np
import cv2



class Patch_creator1(Patch_creator0):
    def __init__(self, grid_size_px, images_informations_preprocessed, test=False):
        super(Patch_creator1, self).__init__(grid_size_px, images_informations_preprocessed, test)

    def transform_back(self, image: np.ndarray, name: str) -> np.ndarray:
        # doc https://docs.opencv.org/master/da/d6e/tutorial_py_geometric_transformations.html
        transformation_matrix = np.array(self.images_informations_preprocessed[name]["transform"])[:-1,:]
        array = cv2.warpAffine(image, transformation_matrix, (*image.shape,)[::-1],)
        return np.array(array)

    def __call__(self, image: np.ndarray, image_name: str, patch_id: int, count_reso=False) -> np.ndarray:
        return super(Patch_creator1, self).__call__(self.transform_back(image,image_name), image_name, patch_id, count_reso)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    FolderInfos.init(test_without_data=True)

    import matplotlib.pyplot as plt
    import os
    import json
    from main.test.test_images import Test_images

    folder = FolderInfos.data_test+"outputs"+FolderInfos.separator+"Patch_creator1"+FolderInfos.separator
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    with open(FolderInfos.data_test+"images_informations_preprocessed.json") as fp: # Load informations about the files
        dico_infos = json.load(fp)

    images_test = Test_images() # Get the object allowing to wuickly get the test images
    array,transform = images_test.get_rasters(selector=0) # get the first test image
    plt.figure() # Create new separated figure
    plt.imshow(array,cmap="gray") #
    plt.savefig(folder+f"{images_test.current_name}_original.png")

    from PIL import Image, ImageDraw
    for grid_size in [500,1000,1500]: # For different grid size we will test creating patches
        patch_creator = Patch_creator1(grid_size_px=grid_size,test=True,images_informations_preprocessed=dico_infos) # Create the path generator object
        plt.clf()
        plt.figure() # Create new separated figure
        img_transf = patch_creator.transform_back(array[:-1,:].astype(np.float32),name=images_test.current_name)
        plt.imshow(img_transf-np.min(img_transf),cmap="gray") #
        plt.savefig(folder+f"{images_test.current_name}_original_transformed_{grid_size}.png")
        for id in range(0,3):#patch_creator.num_available_patches(array)):
            patch = patch_creator(array, images_test.current_name, id) # create the patches specifying additional informations for the statistics
            plt.clf()
            plt.figure()
            plt.title(f"Patch of {grid_size} px length on {images_test.current_name}")
            plt.imshow(patch,cmap="gray",vmin=np.min(array),vmax=np.max(array))
            plt.savefig(folder+f"{images_test.current_name}_patch{id}_size-{grid_size}.png")

        plt.clf() # Clear previous figures
        plt.figure() # Create new separated figure
        # Convert the raster array to a 0-255 array
        image_rgb_uint8 = np.stack((array,)*3,axis=-1)
        image_rgb_uint8 = (image_rgb_uint8 - np.min(image_rgb_uint8)) / (np.max(image_rgb_uint8) - np.min(image_rgb_uint8))*255 # Normalisation
        image_rgb_uint8 = image_rgb_uint8.astype(np.uint8)# Convert to uint8 array
        image_cpy = Image.fromarray(image_rgb_uint8) # Convert to pillow object
        image_cpy1 = Image.fromarray(np.copy(image_rgb_uint8)) # Prepare the image to draw on
        draw = ImageDraw.ImageDraw(image_cpy)
        for coords in patch_creator.coords:
            draw.rectangle(coords, width=30, outline="red")  # Draw the patches
        plt.clf()  # Clear previous figures
        plt.figure()  # Create new separated figure
        image_annotated = Image.blend(image_cpy, image_cpy1, 0.5)  # Mix the original image with the annotated one
        plt.imshow(image_annotated)  # Show it
        plt.title(f"Patches of {grid_size} px length on {images_test.current_name}")
        plt.savefig(
            folder + f"{images_test.current_name}_with_patches_size-{grid_size}.png")  # Save the current figure to file