import json
from typing import Tuple

import numpy as np

from main.FolderInfos import FolderInfos
from main.test.test_images import Test_images
from main.src.param_savers.BaseClass import BaseClass
import time

class Patch_creator0(BaseClass):
    def __init__(self, grid_size_px, images_informations_preprocessed, test=False, exclusion_policy="marginmorethan_1000"):
        self.attr_description = "Create a grid by taking a square of a constant pixel size."+\
                                " It does not consider the resolution of each image."+\
                                " It rejects a patch if it is not fully included in the original image"
        self.attr_grid_size_px = grid_size_px
        self.coords = [] # For logs: to show the square on the original image
        self.test = test
        self.images_informations_preprocessed: dict = images_informations_preprocessed
        self.attr_resolution_used = {"x":{}, "y":{}}
        self.attr_exclusion_policy = exclusion_policy
        self.reject = {}
        self.attr_num_rejected = 0
        self.attr_global_name = "patch_creator"

    def num_available_patches(self,image: np.ndarray ) -> int:
        # return num_lignes*num_cols
        return int(image.shape[0] / self.attr_grid_size_px) * int(image.shape[1] / self.attr_grid_size_px)

    def __call__(self, image: np.ndarray,image_name: str, patch_id: int,count_reso=False, *args, **kargs) -> Tuple[np.ndarray,bool]:
        if count_reso is True: # skiping these lines: 0 ns
            radius_earth_meters = 6371e3
            reso_x = self.images_informations_preprocessed[image_name]["resolution"][0] * np.pi/180. * radius_earth_meters
            reso_y = self.images_informations_preprocessed[image_name]["resolution"][1] * np.pi/180. * radius_earth_meters
            if reso_x not in self.attr_resolution_used["x"].keys():
                self.attr_resolution_used["x"][reso_x] = 0
            if reso_x not in self.attr_resolution_used["y"].keys():
                self.attr_resolution_used["y"][reso_y] = 0
            self.attr_resolution_used["x"][reso_x] += 1
            self.attr_resolution_used["y"][reso_y] += 1

        pos_x,pos_y = self.get_position_patch(patch_id,image.shape) # ~ 0 ns
        if self.test is True: # skipping: 0 ns
            self.coords.append([(pos_y,pos_x),(pos_y + self.attr_grid_size_px,pos_x + self.attr_grid_size_px)])
        patch = image[pos_x:pos_x+self.attr_grid_size_px,pos_y:pos_y+self.attr_grid_size_px] # btwn 8 ms and 26 ms
        # if there are more than x pixels of the patch with the corner value (=0 exactly in float) reject the patch
        # with x the threshold provided in the attr_exclusion_policy after the "_"
        if len(patch[patch == 0]) > int(self.attr_exclusion_policy.split("_")[1]): # 0 ns or 1 ms (sometimes) for the condition. 1 ms and 5 ms for the len(patch.... . 0 ns for the int(.....
            self.attr_num_rejected += 1
            return patch, True
        return patch, False
    def get_position_patch(self,patch_id: int, input_shape):
        num_cols_patches = int(input_shape[1] / self.attr_grid_size_px)
        if num_cols_patches * self.attr_grid_size_px >= input_shape[1]:
            num_cols_patches -= 1
        try:
            id_col = (patch_id) % num_cols_patches
            id_line = patch_id // num_cols_patches
        except:
            raise Exception()
        return self.attr_grid_size_px * id_line,self.attr_grid_size_px * id_col
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
    array,_ = images_test.get_rasters(selector=0) # get the first test image
    plt.figure() # Create new separated figure
    plt.imshow(array,cmap="gray") #
    plt.savefig(folder+f"{images_test.current_name}_original.png")

    from PIL import Image, ImageDraw
    for grid_size in [500,1000,1500]: # For different grid size we will test creating patches
        patch_creator = Patch_creator0(grid_size_px=grid_size,test=True,images_informations_preprocessed=dico_infos) # Create the path generator object

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