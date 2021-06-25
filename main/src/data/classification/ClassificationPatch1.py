from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.data.classification.ClassificationPatch import ClassificationPatch
from main.src.data.patch_creator.patch_creator0 import Patch_creator0
import numpy as np

class ClassificationPatch1(ClassificationPatch):
    """Create and manage patches adding the possibility to use less classes than originally provided

    Args:
        patch_creator: the object of PatchCreator0 class managing patches
        input_size: the size of the image provided as input to the model ⚠️
        limit_num_images: limit the number of image in the dataset per epoch (before filtering)
        classes_to_use: indicates the classes to use in the final classification label
        balance: str enum {nobalance,balance} indicating the class used to balance images
        augmentations_img: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_img: opt str, name of the augmenter to use on source image
        augmentations_patch: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_patch: opt str, name of the augmenter to use on patches
    """
    def __init__(self, patch_creator: Patch_creator0, input_size: int = None, limit_num_images: int = None,
                 classes_to_use="spill,seep", balance="nobalance",
                 augmentations_img="none",augmenter_img="noaugmenter",
                 augmentations_patch="none",augmenter_patch="noaugmenter"):

        self.attr_name = self.__class__.__name__
        super(ClassificationPatch1, self).__init__(patch_creator,input_size,limit_num_images,balance,
                                                   augmentations_img,augmenter_img,augmentations_patch,augmenter_patch)
        tmp_mapping = TwoWayDict({})
        # modifying the class mappings according to attributes provided
        self.attr_classes_to_use = classes_to_use
        for i,name in enumerate(classes_to_use.split(",")):
            tmp_mapping[self.attr_class_mapping[name],Way.ORIGINAL_WAY] = name,i
        self.attr_class_mapping = tmp_mapping
        self.attr_global_name = "dataset"

    def make_classification_label(self, annotations_patch):
        """Creates the classification label based on the annotation patch image

        Indicates if we need to reject the patch due to overrepresented class

        Args:
            annotations_patch: np.ndarray 2d containing for each pixel the class of this pixel

        Returns:
            annotation_modified,reject: the classification label and a boolean to indicate if the patch is rejected or not

        """
        # call the parent method to get classification with parent method make_classification_label
        annotation,reject = super(ClassificationPatch1, self).make_classification_label(annotations_patch)
        # of shape (val_0-1_class_other,val_0-1_class_1,val_0-1_class_2...)
        # Create a classification label with less classes
        annotation_modified = np.zeros((len(self.attr_class_mapping.keys()),))
        # {index_src_0:(class0,index_dest0),index_src_1:(class1,index_dest1),....}
        for src_index,(_,dest_index) in self.attr_class_mapping.items(Way.ORIGINAL_WAY):
             annotation_modified[dest_index] = annotation[src_index]
        return annotation_modified,reject