from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np


class ClassificationPatch2(ClassificationPatch):
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None,
                 classes_to_use="spill,seep", balance="nobalance",margin=None):
        """Creates and manage patches adding the possibility to merge a group of classes as one class telling there is something or not

        Args:
            patch_creator: the object of PatchCreator0 class managing patches
            input_size: the size of the image provided as input to the model ⚠️
            limit_num_images: limit the number of image in the dataset per epoch (before filtering)
            classes_to_use: indicates the classes to use in the final classification label
            balance: str enum {nobalance,balance} indicating the class used to balance images
            margin: opt int, argument for the BalanceClass1 class
        """
        super(ClassificationPatch2, self).__init__(patch_creator, input_size, limit_num_images,balance,margin)
        self.attr_name = self.__class__.__name__
        tmp_mapping = TwoWayDict({})
        self.attr_classes_to_use = classes_to_use
        lkey = []
        lvalue = []
        lname = []
        # merge classes in the dict
        for i, name in enumerate(classes_to_use.split(",")):
            lkey.append(str(self.attr_class_mapping[name]))
            lvalue.append(str(i))
            lname.append(name)
        tmp_mapping["|".join(lkey), Way.ORIGINAL_WAY] = "|".join(lname),"|".join(lvalue)
        self.attr_class_mapping_merged = tmp_mapping
        self.attr_global_name = "dataset"

    def make_classification_label(self, annotations_patch):
        """Creates the classification label based on the annotation patch image

        Merge specified classes together

        Indicates if we need to reject the patch due to overrepresented class

        Args:
            annotations_patch: np.ndarray 2d containing for each pixel the class of this pixel

        Returns:
            annotation_modified,reject: the classification label and a boolean to indicate if the patch is rejected or not

        """
        annotation,reject = super(ClassificationPatch2, self).make_classification_label(annotations_patch)
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        annotation_modified = np.zeros((1,))
        src_indexes = list(map(int,self.attr_class_mapping_merged.keys(Way.ORIGINAL_WAY)[0].split("|")))
        # Merging selected classes together with the max
        for src_index in src_indexes:
            annotation_modified[0] = max(annotation_modified[0],annotation[src_index])
        return annotation_modified,reject