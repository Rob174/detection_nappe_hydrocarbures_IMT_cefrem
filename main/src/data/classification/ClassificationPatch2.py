from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np


class ClassificationPatch2(ClassificationPatch):
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None,classes_to_use="spill,seep", balance="nobalance",margin=None):
        super(ClassificationPatch2, self).__init__(patch_creator, input_size, limit_num_images,balance,margin)
        self.attr_name = self.__class__.__name__
        tmp_mapping = TwoWayDict({})
        self.attr_classes_to_use = classes_to_use
        lkey = []
        lvalue = []
        lname = []
        for i, name in enumerate(classes_to_use.split(",")):
            lkey.append(str(self.attr_class_mapping[name]))
            lvalue.append(str(i))
            lname.append(name)
        tmp_mapping["|".join(lkey), Way.ORIGINAL_WAY] = "|".join(lname),"|".join(lvalue)
        self.attr_class_mapping_merged = tmp_mapping
        self.attr_global_name = "dataset"

    def make_classification_label(self, annotations_patch):
        annotation = super(ClassificationPatch2, self).make_classification_label(annotations_patch)
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        annotation_modified = np.zeros((1,))
        src_indexes = list(map(int,self.attr_class_mapping_merged.keys(Way.ORIGINAL_WAY)[0].split("|")))
        for src_index in src_indexes:
            annotation_modified[0] = max(annotation_modified[0],annotation[src_index])
        return annotation_modified