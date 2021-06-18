import json

import numpy as np

from main.FolderInfos import FolderInfos
from main.test.test_images import Test_images
from main.src.param_savers.BaseClass import BaseClass


class Patch_creator0(BaseClass):
    def __init__(self, grid_size_px, images_informations_preprocessed, test=False, exclusion_policy="marginmorethan_1000"):
        self.attr_description = "Create a grid by taking a square of a constant pixel size."+\
                                " It does not consider the resolution of each image."+\
                                " It rejects a patch if it is not fully included in the original image"
        self.attr_grid_size_px = grid_size_px
        self.coords = [] # For logs: to show the square on the original image
        self.test = test
        self.images_informations_preprocessed: dict = images_informations_preprocessed
        self.resolution_used = {"x":{}, "y":{}}
        self.attr_exclusion_policy = exclusion_policy
        self.reject = False
        self.attr_num_rejected = 0

    def num_available_patches(self,image: np.ndarray ) -> int:
        num_lignes = int(image.shape[0] / self.attr_grid_size_px)
        num_cols = int(image.shape[1] / self.attr_grid_size_px)
        return num_lignes * num_cols

    def __call__(self, image: np.ndarray,image_name: str, patch_id: int,count_reso=False, *args, **kargs) -> np.ndarray:
        if count_reso is True:
            radius_earth_meters = 6371e3
            reso_x = self.images_informations_preprocessed[image_name]["resolution"][0] * np.pi/180. * radius_earth_meters
            reso_y = self.images_informations_preprocessed[image_name]["resolution"][1] * np.pi/180. * radius_earth_meters
            if reso_x not in self.resolution_used["x"].keys():
                self.resolution_used["x"][reso_x] = 0
            if reso_x not in self.resolution_used["y"].keys():
                self.resolution_used["y"][reso_y] = 0
            self.resolution_used["x"][reso_x] += 1
            self.resolution_used["y"][reso_y] += 1

        pos_x,pos_y = self.get_position_patch(patch_id,image.shape)
        if self.test is True:
            self.coords.append([(pos_y,pos_x),(pos_y + self.attr_grid_size_px,pos_x + self.attr_grid_size_px)])
        patch = image[pos_x:pos_x+self.attr_grid_size_px,pos_y:pos_y+self.attr_grid_size_px]
        if patch.shape[0] != self.attr_grid_size_px or patch.shape[1] != self.attr_grid_size_px:
            raise Exception(f"Wrong patch with shape {patch.shape} with id {id}")
        corner_value = 0
        nb_val_out = len(patch[patch == corner_value])
        if nb_val_out > int(self.attr_exclusion_policy.split("_")[1]):
            self.reject = True
            self.attr_num_rejected += 1
        else:
            self.reject = False
        return patch
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
        # for id in range(0,patch_creator.num_available_patches(array)):
        #     patch = patch_creator(array, images_test.current_name, id) # create the patches specifying additional informations for the statistics
        #     plt.clf()
        #     plt.figure()
        #     plt.title(f"Patch of {grid_size} px length on {images_test.current_name}")
        #     plt.imshow(patch,cmap="gray",vmin=np.min(array),vmax=np.max(array))
        #     plt.savefig(folder+f"{images_test.current_name}_patch{id}_size-{grid_size}.png")

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