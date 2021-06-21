from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np

class ClassificationPatch1(ClassificationPatch):
    """Provides data to classify data according to """

    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None,classes_to_use="spill,seep", balance="nobalance",margin=None):
        self.attr_name = self.__class__.__name__
        super(ClassificationPatch1, self).__init__(patch_creator,input_size,limit_num_images,balance,margin)
        tmp_mapping = TwoWayDict({})
        self.attr_classes_to_use = classes_to_use
        for i,name in enumerate(classes_to_use.split(",")):
            tmp_mapping[self.attr_class_mapping[name],Way.ORIGINAL_WAY] = name,i
        self.attr_class_mapping = tmp_mapping
        self.attr_global_name = "dataset"

    def make_classification_label(self, annotations_patch):
        annotation,reject = super(ClassificationPatch1, self).make_classification_label(annotations_patch)
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        annotation_modified = np.zeros((len(self.attr_class_mapping.keys(),)))
        # {index_src_0:(class0,index_dest0),index_src_1:(class1,index_dest1),....}
        for src_index,(_,dest_index) in self.attr_class_mapping.items(Way.ORIGINAL_WAY):
             annotation_modified[dest_index] = annotation[src_index]
        return annotation_modified,reject