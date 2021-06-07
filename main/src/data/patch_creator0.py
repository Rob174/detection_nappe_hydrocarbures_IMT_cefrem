import json
from typing import List

import numpy as np

from main.FolderInfos import FolderInfos
from main.test.test_images import Test_images


class Patch_creator0:
    def __init__(self, grid_size_px,test=False):
        self.attr_description = "Create a grid by taking a square of a constant pixel size."+\
                                " It does not consider the resolution of each image."+\
                                " It rejects a patch if it is not fully included in the original image"
        self.attr_grid_size_px = grid_size_px
        self.attr_stat_patch_meters_size = {"x":{}, "y":{}}
        self.coords = [] # For logs: to show the square on the original image
        self.test = test
    def __call__(self, image: np.ndarray,image_name:str,images_informations_preprocessed: dict) -> List[np.ndarray]:
        resolution_x = images_informations_preprocessed[image_name]["resolution"][0] * self.attr_grid_size_px
        resolution_y = images_informations_preprocessed[image_name]["resolution"][1] * self.attr_grid_size_px
        if resolution_x not in self.attr_stat_patch_meters_size["x"].keys():
            self.attr_stat_patch_meters_size["x"][resolution_x] = 0
        if resolution_y not in self.attr_stat_patch_meters_size["y"].keys():
            self.attr_stat_patch_meters_size["y"][resolution_y] = 0
        self.attr_stat_patch_meters_size["x"][resolution_x] += 1
        self.attr_stat_patch_meters_size["y"][resolution_y] += 1

        list_patches: List[np.ndarray] = []
        for x in np.arange(0, image.shape[0], self.attr_grid_size_px).astype(int):
            for y in np.arange(0, image.shape[1], self.attr_grid_size_px).astype(int):
                if x + self.attr_grid_size_px <= image.shape[0] and y + self.attr_grid_size_px <= image.shape[1]:
                    # If the patch is not fully include in the original image
                    list_patches.append(image[x:x + self.attr_grid_size_px, y:y + self.attr_grid_size_px])
                    if self.test is True:
                        self.coords.append([(y,x),(y + self.attr_grid_size_px, x + self.attr_grid_size_px)])

        return list_patches

if __name__ == "__main__":

    FolderInfos.init(test_without_data=True)

    import matplotlib.pyplot as plt
    import os

    folder = FolderInfos.data_test+"outputs"+FolderInfos.separator+"Patch_creator"+FolderInfos.separator
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    with open(FolderInfos.data_test+"images_informations_preprocessed.json") as fp: # Load informations about the files
        dico_infos = json.load(fp)

    images_test = Test_images() # Get the object allowing to wuickly get the test images
    array = images_test.get_rasters(selector=0) # get the first test image
    plt.figure() # Create new separated figure
    plt.imshow(array,cmap="gray") #
    plt.savefig(folder+f"{images_test.current_name}_original.png")

    from PIL import Image, ImageDraw
    for grid_size in [500,1000,1500]: # For different grid size we will test creating patches
        patch_creator = Patch_creator0(grid_size_px=grid_size,test=True) # Create the path generator object
        patches = patch_creator(array, images_test.current_name, dico_infos) # create the patches specifying additional informations for the statistics

        plt.clf() # Clear previous figures
        plt.figure() # Create new separated figure
        # Convert the raster array to a 0-255 array
        image_rgb_uint8 = np.stack((array,)*3,axis=-1)*255
        image_rgb_uint8 = (image_rgb_uint8 - np.min(image_rgb_uint8)) / (np.max(image_rgb_uint8) - np.min(image_rgb_uint8)) # Normalisation
        image_rgb_uint8 = image_rgb_uint8.astype(np.uint8)# Convert to uint8 array
        image_cpy = Image.fromarray(image_rgb_uint8) # Convert to pillow object
        image_cpy1 = Image.fromarray(np.copy(image_rgb_uint8)) # Prepare the image to draw on
        draw = ImageDraw.ImageDraw(image_cpy)
        for coords in patch_creator.coords:
            draw.rectangle(coords,width=30,outline="red") # Draw the patches
        plt.clf() # Clear previous figures
        plt.figure()# Create new separated figure
        image_annotated = Image.blend(image_cpy, image_cpy1, 0.5) # Mix the original image with the annotated one
        plt.imshow(image_annotated) # Show it
        plt.title(f"Patches of {grid_size} px length on {images_test.current_name}")
        plt.savefig(folder+f"{images_test.current_name}_with_patches_size-{grid_size}.png") # Save the current figure to file
        for i,patch in enumerate(patches):
            plt.clf()
            plt.figure()
            plt.title(f"Patch of {grid_size} px length on {images_test.current_name}")
            plt.imshow(patch,cmap="gray",vmin=np.min(array),vmax=np.max(array))
            plt.savefig(folder+f"{images_test.current_name}_patch{i}_size-{grid_size}.png")