from typing import List

from main.src.data.classification.ClassificationPatch import ClassificationPatch
import numpy as np

class VectorizzedClassificationPatch(ClassificationPatch):
    """Class that adapt the inputs from the hdf5 file (input image, label image), and manage other objects to create patches,
    filter them.

    It is a specialized version that allows to process multiple items together

    Args:
        patch_creator: the object of PatchCreator0 class managing patches
        input_size: the size of the image provided as input to the model ⚠️
        limit_num_images: limit the number of image in the dataset per epoch (before filtering)
        balance: str enum {nobalance,balance} indicating the class used to balance images
        margin: opt int, argument for the BalanceClass1 class
        augmentations_img: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_img: opt str, name of the augmenter to use on source image
        augmentations_patch: opt str, list of augmentations to apply separated by commas to apply to source image
        augmenter_patch: opt str, name of the augmenter to use on patches
    """
    def __init__(self,*args,**kargs):
        super(VectorizzedClassificationPatch, self).__init__(*args,**kargs)
    def getitem(self, id: List[int]):
        """Magic method of python called by the object[id] syntax.

        get the patch of global int ids id

        Args:
            id: list of int, global ⚠️ id of the patch

        Returns:
            tuple of 3 lists :

            - patches:   np.ndarray (shape (grid_size,grid_size,3)), input image for the model
            - annotations: np.ndarray (shape (num_classes,), classification patch
            - rejects:  bool, indicate if we need to reject this sample

        """
        raise Exception("Not adapted")
        all_items = self.get_all_items()
        dico_item_patches = {}
        for index_orig, i in enumerate(id):
            item_name, patch_id = all_items[i]
            if item_name not in dico_item_patches:
                dico_item_patches[item_name] = []
            dico_item_patches[item_name].append([patch_id, index_orig])
        dico_values_patches = {}
        dico_values_patch_annotations = {}
        dico_values_patch_rejects = {}
        for item_name, liste_patches in dico_item_patches.items():
            img = self.getimage(item_name)
            annotations = self.annotations_labels[item_name]
            # Make augmentations if necessary (thanks to NoAugment class
            img,annotations = self.attr_img_augmenter.transform(img,annotations)
            # from here can be parallellized (gpu ??)
            patches_ids = list(map(lambda x: x[0], liste_patches))
            for patch_id, original_index in patches_ids:
                img_patch, reject = self.patch_creator(img, item_name, patch_id=patch_id)
                annotations_patch, reject = self.patch_creator(annotations, item_name, patch_id=patch_id)
                if reject is False:
                    img_patch, annotations_patch = self.attr_patch_augmenter.transform(img_patch, annotations_patch)
                dico_values_patches[original_index] = img_patch
                dico_values_patch_annotations[original_index] = annotations_patch
                dico_values_patch_rejects[original_index] = reject
        patches = np.stack([np.stack((dico_values_patches[k],)*3,axis=0) for k in sorted(dico_values_patches.keys())],axis=0)
        annotations = np.stack([self.make_classification_label(dico_values_patch_annotations[k]) for k in sorted(dico_values_patch_annotations.keys())],axis=0)
        rejects = np.array([dico_values_patch_rejects[k] for k in sorted(dico_values_patch_rejects.keys())])
        return patches,annotations,rejects