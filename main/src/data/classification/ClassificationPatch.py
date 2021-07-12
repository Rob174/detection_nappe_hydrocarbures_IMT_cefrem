import random
from typing import Optional
from typing import Tuple

import numpy as np
from rasterio.transform import Affine, rowcol

from main.src.data.Augmentation.Augmenters.Augmenter0 import Augmenter0
from main.src.data.Augmentation.Augmenters.Augmenter1 import Augmenter1
from main.src.data.Augmentation.Augmenters.NoAugmenter import NoAugmenter
from main.src.data.Augmentation.Augmenters.enums import EnumAugmenter
from main.src.data.TwoWayDict import Way
from main.src.data.balance_classes.balance_classes import BalanceClasses1
from main.src.data.balance_classes.enums import EnumBalance
from main.src.data.balance_classes.no_balance import NoBalance
from main.src.data.balance_classes.only_other import BalanceClasses2
from main.src.data.classification.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.classification.LabelModifier.LabelModifier2 import LabelModifier2
from main.src.data.classification.LabelModifier.NoLabelModifier import NoLabelModifier
from main.src.data.classification.enums import EnumLabelModifier
from main.src.data.enums import EnumClasses
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
from main.src.data.resizer import Resizer
from main.src.data.segmentation.DataSentinel1Segmentation import DataSentinel1Segmentation


class ClassificationPatch(DataSentinel1Segmentation):
    """Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches,
    filteer them.

    Args:
        patch_creator: the object of PatchCreator0 class managing patches
        input_size: the size of the image provided as input to the attr_model ⚠️
        limit_num_images: limit the number of image in the attr_dataset per epoch (before filtering)
        balance: EnumBalance indicating the class used to balance images
        augmentations_img: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_img: opt EnumAugmenter, name of the augmenter to use on source image
        augmentations_patch: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_patch: opt EnumAugmenter, name of the augmenter to use on patches
        augmentation_factor: the number of replicas of the original attr_dataset to do
        label_modifier: EnumLabelModifier
    """

    def __init__(self, patch_creator: Patch_creator0, input_size: int = None,
                 limit_num_images: int = None, balance: EnumBalance = EnumBalance.NoBalance,
                 augmentations_img="none", augmenter_img: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentations_patch="none", augmenter_patch: EnumAugmenter = EnumAugmenter.NoAugmenter,
                 augmentation_factor: int = 100, label_modifier: EnumLabelModifier = EnumLabelModifier.NoLabelModifier,
                 classes_to_use: Tuple[EnumClasses] = (EnumClasses.Seep, EnumClasses.Spill),
                 tr_percent=0.7):
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.patch_creator = patch_creator
        self.attr_limit_num_images = limit_num_images
        self.attr_resizer = Resizer(out_size_w=input_size)
        self.attr_augmentation_factor = augmentation_factor
        super(ClassificationPatch, self).__init__(limit_num_images, input_size=input_size)
        self.tr_keys = list(self.images.keys())[:int(len(self.images) * tr_percent)]
        self.valid_keys = list(self.images.keys())[int(len(self.images) * tr_percent):]
        self.attr_global_name = "attr_dataset"
        if label_modifier == EnumLabelModifier.NoLabelModifier:
            self.attr_label_modifier = NoLabelModifier()
        elif label_modifier == EnumLabelModifier.LabelModifier1:
            self.attr_label_modifier = LabelModifier1(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        elif label_modifier == EnumLabelModifier.LabelModifier2:
            self.attr_label_modifier = LabelModifier2(classes_to_use=classes_to_use,
                                                      original_class_mapping=DataSentinel1Segmentation.attr_original_class_mapping)
        else:
            raise NotImplementedError(f"{label_modifier} is not implemented")
        if balance == EnumBalance.BalanceClasses1:
            self.attr_balance = NoBalance()
        elif balance == EnumBalance.NoBalance:
            # see class DataSentinel1Segmentation for documentation on attr_class_mapping storage and access to values
            self.attr_balance = BalanceClasses1(other_index=self.attr_original_class_mapping["other"])
        elif balance == EnumBalance.NoBalance:
            self.attr_balance = BalanceClasses2(other_index=self.attr_original_class_mapping["other"])
        if augmentations_img != "none":
            if augmenter_img == EnumAugmenter.Augmenter0:
                self.attr_img_augmenter = Augmenter0(allowed_transformations=augmentations_img)
                self.generator = self.generate_item_step_by_step
            elif augmenter_img == EnumAugmenter.Augmenter1:
                self.attr_img_augmenter = Augmenter1(allowed_transformations=augmentations_img,
                                                     patch_size_before_final_resize=
                                                     self.patch_creator.attr_grid_size_px,
                                                     patch_size_final_resize=input_size
                                                     )
                self.generator = self.generate_item_with_augmentation_at_once

            else:
                self.generator = self.generate_item_step_by_step
                raise NotImplementedError(f"{augmenter_img} is not implemented")
        else:
            self.generator = self.generate_item_step_by_step
            self.attr_img_augmenter = NoAugmenter()
        if augmentations_patch != "none":
            if augmenter_patch == EnumAugmenter.Augmenter0:
                self.attr_patch_augmenter = Augmenter0(allowed_transformations=augmentations_patch)
            else:
                raise NotImplementedError(f"{augmenter_patch} is not implemented")
        else:
            self.attr_patch_augmenter = NoAugmenter()
        # Cache to store between epochs rejected images if we have no image augmenter
        self.cache_img_id_rejected = []

    def get_geographic_coords_of_patch(self, name_src_img, patch_id):
        """Get the coordinates of the upper left pixel of the patch specified

        Args:
            name_src_img: str, uniq id of the source image
            patch_id: int, id of the patch to get coordinates

        Returns:
            tuple xcoord,ycoord coordinates of the upper left pixel of the patch specified
        """
        img = self.getimage(name_src_img)  # read image from the hdf5 file
        transform_array = self.images_infos[name_src_img]["transform"]  # get the corresponding transformation array
        transform_array = np.array(transform_array)
        # transfer it into a rasterio AffineTransformation object
        transform = Affine.from_gdal(a=transform_array[0, 0], b=transform_array[0, 1], c=transform_array[0, 2],
                                     d=transform_array[1, 0], e=transform_array[1, 1], f=transform_array[1, 2])
        # Get the position of the upperleft pixel on the global image
        posx, posy = self.patch_creator.get_position_patch(patch_id=patch_id, input_shape=img.shape)
        # Get the corresponding geographical coordinates
        return rowcol(transform, posx, posy)

    def __iter__(self, dataset="tr"):
        return iter(self.generator(dataset))

    def generate_item_with_augmentation_at_once(self, dataset="tr"):
        """

        Args:
            dataset: str, tr or valid to choose source images for tr or valid attr_dataset

        Returns:
            generator of the attr_dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the attr_model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   transformation_matrix:  the transformation matrix to transform the source image
                   item: str name of the source image
        """
        if isinstance(self.attr_img_augmenter, Augmenter1) is False:
            raise Exception("Only augmenter1 is supported with this method of attr_dataset generation")
        if isinstance(self.attr_patch_augmenter, NoAugmenter) is False:
            raise Exception("The patch augmenter is not supported when you choose augmenter1 for image augmenter")
        images_available = self.tr_keys if dataset == "tr" else self.valid_keys
        for num_dataset in range(self.attr_augmentation_factor):
            random.shuffle(images_available)
            for item in images_available:
                image = np.array(self.images[item])
                annotations = np.array(self.annotations_labels[item], dtype=np.float32)
                partial_transformation_matrix = self.attr_img_augmenter.choose_new_augmentations(image)
                for patch_upper_left_corner_coords in np.random.permutation(
                        self.attr_img_augmenter.get_grid(image.shape, partial_transformation_matrix)):
                    image_patch, annotations_patch, transformation_matrix = self.attr_img_augmenter.transform(
                        image=image,
                        annotation=annotations,
                        partial_transformation_matrix=partial_transformation_matrix,
                        patch_upper_left_corner_coords=patch_upper_left_corner_coords
                        )
                    reject = self.patch_creator.check_reject(image_patch, threshold_px=10)
                    if reject is True:
                        # print("reject")
                        continue
                    # Create the classification label with the proper technic ⚠️⚠️ inheritance
                    classification, balance_reject = self.make_classification_label(annotations_patch)  # ~ 2 ms
                    if balance_reject is True:
                        # print("reject due to balance ",classification)
                        continue
                    # convert the image to rgb (as required by pytorch): not ncessary the best transformation as we multiply by 3 the amount of data
                    image_patch = np.stack((image_patch, image_patch, image_patch), axis=0)  # 0 ns most of the time
                    yield image_patch, annotations, transformation_matrix, item

    def generate_item_step_by_step(self, dataset="tr"):  # btwn 25 and 50 ms
        """Magic method of python called by the object[id] syntax.

        get the patch of global int id id
        Args:
            dataset:

        Returns:
            generator of the attr_dataset (object that support __iter__ and __next__ magic methods)
            tuple: input: np.ndarray (shape (grid_size,grid_size,3)), input image for the attr_model ;
                   classif: np.ndarray (shape (num_classes,), classification patch ;
                   None:  no transformation matrix is available for this method
                   item: str name of the source image
        """
        if isinstance(self.attr_img_augmenter, Augmenter1) is True:
            raise Exception("Augmenter1 is not supported with this method of attr_dataset generation. Use Augmenter0")
        images_available = self.tr_keys if dataset == "tr" else self.valid_keys
        for num_dataset in range(self.attr_augmentation_factor):
            random.shuffle(images_available)
            for item in images_available:
                image = self.images[item]
                annotations = self.annotations_labels[item]
                for patch_id in np.random.permutation(range(self.patch_creator.num_available_patches(image))):

                    if isinstance(self.attr_img_augmenter, NoAugmenter) and \
                            [item, patch_id] in self.cache_img_id_rejected:
                        continue  # to save computation time and keep np array output
                    # Make augmentations on input image if necessary (thanks to NoAugment class)
                    image, annotations = self.attr_img_augmenter.transform(image, annotations)
                    # get the patch with the selected id for the input image and the annotation
                    ## two lines: btwn 21 and 54 ms
                    img_patch, reject = self.patch_creator(image, item, patch_id=patch_id)  # btwn 10 ms and 50 ms
                    if reject is True:
                        continue  # to save computation time and keep np array output
                    annotations_patch, _ = self.patch_creator(annotations, item,
                                                              patch_id=patch_id)  # btwn 10 ms and 30 ms (10 ms most of the time)
                    # Make augmentations on patch if necessary (thanks to NoAugment class) and if it is not rejected
                    img_patch, annotations_patch = self.attr_patch_augmenter.transform(img_patch, annotations_patch)
                    # we reject an image if it contains margins (cf patchcreator)
                    # resize the image at the provided size in the constructor (with the magic method __call__ of the Resizer object
                    input = self.attr_resizer(img_patch)  # ~ 0 ns most of the time, 1 ms sometimes
                    # convert the image to rgb (as required by pytorch): not ncessary the best transformation as we multiply by 3 the amount of data
                    input = np.stack((input, input, input), axis=0)  # 0 ns most of the time
                    input = (input - self.pixel_stats["mean"]) / self.pixel_stats["std"]
                    # Create the classification label with the proper technic ⚠️⚠️ inheritance
                    classif, balance_reject = self.make_classification_label(annotations_patch)  # ~ 2 ms
                    # As the balancing operation are done in the make_classification_label method, we reject an image
                    # if it is rejected due to margins or balancing
                    reject = reject or balance_reject
                    if isinstance(self.attr_img_augmenter, NoAugmenter) and reject is True:
                        self.cache_img_id_rejected.append([item, patch_id])
                        continue
                    yield input, classif, None, item

    def make_classification_label(self, annotations_patch):
        """Creates the classification label based on the annotation patch image

        Indicates if we need to reject the patch due to overrepresented class

        Args:
            annotations_patch: np.ndarray 2d containing for each pixel the class of this pixel

        Returns: the classification label

        """

        output = np.zeros((len(self.attr_original_class_mapping),), dtype=np.float32)  # 0 ns
        for value in self.attr_original_class_mapping.keys(Way.ORIGINAL_WAY):  # btwn 1 and 2 ms for the for loop
            # for each class of the original attr_dataset, we put a probability of presence of one if the class is in the patch
            value = int(value)
            #  if the class is in the patch
            if value in annotations_patch:
                output[value] = 1.
        # Check if we need to reject the patch due to overrepresented class
        balance_reject = self.attr_balance.filter(output)
        # Modify the label if require
        output = self.attr_label_modifier.make_classification_label(output)
        return output, balance_reject

    def __len__(self):
        return None

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
        last_image = np.copy(np.array(self.getimage(name), dtype=np.float32))
        liste_patches = []
        num_patches = self.patch_creator.num_available_patches(last_image)
        # Create all the patches of input images
        for id in range(num_patches):
            patch, reject = self.patch_creator(last_image, name, patch_id=id)
            liste_patches.append([patch])
            liste_patches[id].append(reject)
        annotations = np.array(self.annotations_labels[name], dtype=np.float32)
        for id in range(num_patches):
            patch, reject = self.patch_creator(annotations, name, patch_id=id)
            classif = self.make_classification_label(patch)
            # we ignore balancing rejects
            liste_patches[id].insert(1, classif[0])
        return liste_patches

    def len(self, dataset: str) -> Optional[int]:
        return None
# Tests in DatasetFactory
