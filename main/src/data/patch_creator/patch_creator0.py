import json
from typing import Tuple

import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.patch_creator.enums import EnumPatchExcludePolicy
from main.src.param_savers.BaseClass import BaseClass
from main.test.test_images import Test_images


class Patch_creator0(BaseClass):
    """Class creating and managing patches

    Args:
        grid_size_px: size of the patch extracted from the original image
        images_informations_preprocessed: dict from the json file containing images attr_dataset informations
        test: bool, indicate if we need to keep track of the coordinates (in px) of the patch computed
        exclusion_policy: EnumPatchExcludePolicy, policy used to exclude patches based on their apearence.
        exclusion_policy_threshold: for EnumPatchExcludePolicy.MarginMoreThan number of pixels at 0 exactly after which a patch is excluded
    """

    def __init__(self, grid_size_px, images_informations_preprocessed, test=False,
                 exclusion_policy: EnumPatchExcludePolicy = EnumPatchExcludePolicy.MarginMoreThan,
                 exclusion_policy_threshold: int = 1000):
        self.attr_description = "Create a grid by taking a square of a constant pixel size." + \
                                " It does not consider the resolution of each image." + \
                                " It rejects a patch if it is not fully included in the original image"
        self.attr_grid_size_px = grid_size_px
        self.coords = []  # For logs: to show the square on the original image
        self.test = test
        self.images_informations_preprocessed: dict = images_informations_preprocessed
        self.attr_resolution_used = {"x": {}, "y": {}}  # dict of dict to count the number of uniq resolution seen
        self.attr_exclusion_policy = exclusion_policy
        self.attr_exclusion_policy_threshold = exclusion_policy_threshold
        self.reject = {}
        self.attr_num_rejected = 0
        self.attr_name = self.__class__.__name__
        self.attr_global_name = "patch_creator"

    def num_available_patches(self, image: np.ndarray) -> int:
        """

        Args:
            image: original image of the hdf5 file

        Returns:
            number of patches possible without cutting any of them

        """
        return int(image.shape[0] / self.attr_grid_size_px) * int(image.shape[1] / self.attr_grid_size_px)

    def __call__(self, image: np.ndarray, image_name: str, patch_id: int, count_reso=False):
        return self.call(image, image_name, patch_id, count_reso)

    def call(self, image: np.ndarray, image_name: str, patch_id: int, count_reso=False) -> Tuple[np.ndarray, bool]:
        """Creates the patch with provided data and indicate if it is rejected

        Magic method __call__ "called" when we "evaluate" the object:

        obj = MyClass(....); returnValue = obj(...) <- this call the __call__ magic method

        Args:
            image: np.ndarray, source image from the hdf5 file
            image_name: str, uniq id of the image in the hdf5 cache
            patch_id: int, id of the patch
            count_reso: bool, indicate if we need to compute statistics on input image resolutions

        Returns:
            tuple of np.ndarray, the patch and a boolean indicating if the patch is rejected

        """
        # computing resolution statistics taking a attr_model where the earth is a sphere
        if count_reso is True:  # skiping these lines: 0 ns
            radius_earth_meters = 6371e3
            reso_x = self.images_informations_preprocessed[image_name]["resolution"][
                         0] * np.pi / 180. * radius_earth_meters
            reso_y = self.images_informations_preprocessed[image_name]["resolution"][
                         1] * np.pi / 180. * radius_earth_meters
            if reso_x not in self.attr_resolution_used["x"].keys():
                self.attr_resolution_used["x"][reso_x] = 0
            if reso_x not in self.attr_resolution_used["y"].keys():
                self.attr_resolution_used["y"][reso_y] = 0
            self.attr_resolution_used["x"][reso_x] += 1
            self.attr_resolution_used["y"][reso_y] += 1

        pos_x, pos_y = self.get_position_patch(patch_id, image.shape)  # ~ 0 ns
        # Save the coordinates (in px on the image from the hdf5) of the patch if we need them (especially during tests)
        if self.test is True:  # skipping: 0 ns
            self.coords.append([(pos_y, pos_x), (pos_y + self.attr_grid_size_px, pos_x + self.attr_grid_size_px)])
        # extract the patch
        patch = image[pos_x:pos_x + self.attr_grid_size_px, pos_y:pos_y + self.attr_grid_size_px]  # btwn 8 ms and 26 ms
        # if there are more than x pixels of the patch with the corner value (=0 exactly in float) reject the patch
        # with x the threshold provided in the attr_exclusion_policy after the "_"
        if Patch_creator0.check_reject(patch,
                                       self.attr_exclusion_policy_threshold) is True:  # 0 ns or 1 ms (sometimes) for the condition. 1 ms and 5 ms for the len(patch.... . 0 ns for the int(.....
            self.attr_num_rejected += 1
            return patch, True
        else:
            return patch, False

    @staticmethod
    def check_reject(patch: np.ndarray, threshold_px: int):
        if len(patch[
                   patch == 0]) > threshold_px:  # 0 ns or 1 ms (sometimes) for the condition. 1 ms and 5 ms for the len(patch.... . 0 ns for the int(.....
            return True
        return False

    def get_position_patch(self, patch_id: int, input_shape):
        """Compute the position of the patch with respect to the parameters

        Args:
            patch_id: int,
            input_shape: tuple with at least two values

        Returns:
            tuple of ints: xpos,ypos
        """
        num_cols_patches = int(input_shape[1] / self.attr_grid_size_px)
        if num_cols_patches * self.attr_grid_size_px >= input_shape[1]:
            num_cols_patches -= 1
        try:
            id_col = (patch_id) % num_cols_patches
            id_line = patch_id // num_cols_patches
        except:
            raise Exception()
        return self.attr_grid_size_px * id_line, self.attr_grid_size_px * id_col


if __name__ == "__main__":

    FolderInfos.init(test_without_data=True)

    import matplotlib.pyplot as plt
    import os

    folder = FolderInfos.data_test + "outputs" + FolderInfos.separator + "Patch_creator" + FolderInfos.separator
    if os.path.exists(folder) is False:
        os.mkdir(folder)
    with open(
            FolderInfos.data_test + "images_informations_preprocessed.json") as fp:  # Load informations about the files
        dico_infos = json.load(fp)

    images_test = Test_images()  # Get the object allowing to wuickly get the test images
    array, _ = images_test.get_rasters(selector=0)  # get the first test image
    plt.figure()  # Create new separated figure
    plt.imshow(array, cmap="gray")  #
    plt.savefig(folder + f"{images_test.current_name}_original.png")

    from PIL import Image, ImageDraw

    for grid_size in [500, 1000, 1500]:  # For different grid size we will test creating patches
        patch_creator = Patch_creator0(grid_size_px=grid_size, test=True,
                                       images_informations_preprocessed=dico_infos)  # Create the path generator object

        plt.clf()  # Clear previous figures
        plt.figure()  # Create new separated figure
        # Convert the raster array to a 0-255 array
        image_rgb_uint8 = np.stack((array,) * 3, axis=-1)
        image_rgb_uint8 = (image_rgb_uint8 - np.min(image_rgb_uint8)) / (
                    np.max(image_rgb_uint8) - np.min(image_rgb_uint8)) * 255  # Normalisation
        image_rgb_uint8 = image_rgb_uint8.astype(np.uint8)  # Convert to uint8 array
        image_cpy = Image.fromarray(image_rgb_uint8)  # Convert to pillow object
        image_cpy1 = Image.fromarray(np.copy(image_rgb_uint8))  # Prepare the image to draw on
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
