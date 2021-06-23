from functools import lru_cache
from typing import Tuple, List, Union
import numpy as np
import psutil

from main.FolderInfos import FolderInfos
from main.src.data.TwoWayDict import  Way
from main.src.data.balance_classes.balance_classes import BalanceClasses1
from main.src.data.balance_classes.no_balance import NoBalance
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation
import time
from rasterio.transform import Affine,rowcol


class ClassificationPatch(DataSentinel1Segmentation):
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None,
                 limit_num_images: int = None, balance="nobalance",margin=None):
        """Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches,
        filteer them.

        Args:
            patch_creator: the object of PatchCreator0 class managing patches
            input_size: the size of the image provided as input to the model ⚠️
            limit_num_images: limit the number of image in the dataset per epoch (before filtering)
            balance: str enum {nobalance,balance} indicating the class used to balance images
            margin: opt int, argument for the BalanceClass1 class
        """
        self.attr_name = self.__class__.__name__ # save the name of the class used for reproductibility purposes
        self.patch_creator = patch_creator
        self.attr_limit_num_images = limit_num_images
        self.attr_resizer = Resizer(out_size_w=input_size)
        super(ClassificationPatch, self).__init__(limit_num_images, input_size=input_size)
        self.attr_global_name = "dataset"
        if balance == "nobalance":
            self.attr_balance = NoBalance()
        elif balance == "balanceclasses1":
            # see class DataSentinel1Segmentation for documentation on attr_class_mapping storage and access to values
            self.attr_balance = BalanceClasses1(classes_indexes=self.attr_class_mapping.keys(Way.ORIGINAL_WAY),
                                                margin=margin)


    @lru_cache(maxsize=1)
    def get_all_items(self):
        """List available original images available in the dataset (hdf5 file)
        the :lru_cache(maxsize=1) allow to compute it only one time and store the result in a cache

        Allow to limit the number of original image used in the dataset

        Returns: list of list of str,int: [[img_uniq_id,id_patch],...]

        """
        list_items = []
        for img_name in list(self.images.keys()):
            img = self.getitem(img_name)
            num_ids = self.patch_creator.num_available_patches(img)
            list_items.extend([[img_name, i] for i in range(num_ids)])
        if self.attr_limit_num_images is not None: # Limit the number of images used
            return list_items[:self.attr_limit_num_images]
        return list_items
    def get_geographic_coords_of_patch(self,name_src_img,patch_id):
        """Get the coordinates of the upper left pixel of the patch specified

        Args:
            name_src_img: str, uniq id of the source image
            patch_id: int, id of the patch to get coordinates

        Returns:
            tuple xcoord,ycoord coordinates of the upper left pixel of the patch specified
        """
        img = self.getitem(name_src_img) # read image from the hdf5 file
        transform_array = self.images_infos[name_src_img]["transform"] # get the corresponding transformation array
        transform_array = np.array(transform_array)
        # transfer it into a rasterio AffineTransformation object
        transform = Affine.from_gdal(a=transform_array[0, 0], b=transform_array[0, 1], c=transform_array[0, 2],
                                     d=transform_array[1, 0], e=transform_array[1, 1], f=transform_array[1, 2])
        # Get the position of the upperleft pixel on the global image
        posx,posy = self.patch_creator.get_position_patch(patch_id=patch_id,input_shape=img.shape)
        # Get the corresponding geographical coordinates
        return rowcol(transform,posx,posy)
    def __getitem__(self, id: Union[int,List[int]]) -> Tuple[np.ndarray, np.ndarray,bool]:
        return self.getitem(id)
    def getitem(self, id: Union[int,List[int]]) -> Tuple[np.ndarray, np.ndarray,bool]: # btwn 25 and 50 ms
        """Magic method of python called by the object[id] syntax.

        get the patch of global int id id

        Args:
            id: int, global ⚠️ id of the patch

        Returns:
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   reject:  bool, indicate if we need to reject this sample ;
        """
        # get the src image id (item: str) and the patch_id (int)
        [item, patch_id] = self.get_all_items()[id] # 0 ns
        # get the source image from the hdf5 cache
        img = self.getitem(item) # 1ms but 0 most of the time
        # get the source true classification / annotation from the other hdf5 cache
        annotations = self.annotations_labels[item] # 1ms but 0 most of the time
        # get the patch with the selected id for the input image and the annotation
        ## two lines: btwn 21 and 54 ms
        img_patch,reject = self.patch_creator(img, item, patch_id=patch_id) # btwn 10 ms and 50 ms
        annotations_patch,reject = self.patch_creator(annotations, item, patch_id=patch_id) # btwn 10 ms and 30 ms (10 ms most of the time)
        # we reject an image if it contains margins (cf patchcreator)
        # resize the image at the provided size in the constructor (with the magic method __call__ of the Resizer object
        input = self.attr_resizer(img_patch) # ~ 0 ns most of the time, 1 ms sometimes
        # convert the image to rgb (as required by pytorch): not ncessary the best transformation as we multiply by 3 the amount of data
        input = np.stack((input, input, input), axis=0) # 0 ns most of the time
        # Create the classification label with the proper technic ⚠️⚠️ inheritance
        classif,balance_reject = self.make_classification_label(annotations_patch) # ~ 2 ms
        # As the balancing operation are done in the make_classification_label method, we reject an image
        # if it is rejected due to margins or balancing
        reject = reject and balance_reject
        return input, classif, reject

    def make_classification_label(self, annotations_patch):
        """Creates the classification label based on the annotation patch image

        Indicates if we need to reject the patch due to overrepresented class

        Args:
            annotations_patch: np.ndarray 2d containing for each pixel the class of this pixel

        Returns: the classification label

        """

        output = np.zeros((len(self.attr_original_class_mapping),),dtype=np.float32) # 0 ns
        for value in self.attr_original_class_mapping.keys(Way.ORIGINAL_WAY): # btwn 1 and 2 ms for the for loop
            # for each class of the original dataset, we put a probability of presence of one if the class is in the patch
            value = int(value)
            #  if the class is in the patch
            if value in annotations_patch:
                output[value] = 1.
        # Check if we need to reject the patch due to overrepresented class
        balance_reject = self.attr_balance.filter(output)
        return output,balance_reject

    def make_patches_of_image(self, name: str):
        """Creates and returns all patches of an image

        Args:
            name: uniq str id of the image

        Returns:
            list of list of:

            - patch: np.ndarray
            - classif: np.ndarray classification label as returned by make_classification_label
            - reject: bool reject only based on margins
        """
        last_image = np.copy(np.array(self.getitem(name), dtype=np.float32))
        liste_patches = []
        num_patches = self.patch_creator.num_available_patches(last_image)
        # Create all the patches of input images
        for id in range(num_patches):
            patch,reject = self.patch_creator(last_image, name, patch_id=id, keep=True)
            liste_patches.append([patch])
        annotations = np.array(self.annotations_labels[name], dtype=np.float32)
        for id in range(num_patches):
            patch,reject = self.patch_creator(annotations, name, patch_id=id)
            classif = self.make_classification_label(patch)
            # we ignore balancing rejects
            liste_patches[id].append(classif[0])
            liste_patches[id].append(reject)
        return liste_patches

    def __len__(self) -> int:
        """Magic method called when we make len(obj)"""
        return len(self.get_all_items())

# Tests in DatasetFactory
