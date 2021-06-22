from typing import List

from main.src.data.classification.ClassificationPatch import ClassificationPatch
import numpy as np

class VectorizzedClassificationPatch(ClassificationPatch):
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
            img = self.images[item_name]
            annotations = self.annotations_labels[item_name]
            # from here can be parallellized (gpu ??)
            patches_ids = list(map(lambda x: x[0], liste_patches))
            for patch_id, original_index in patches_ids:
                img_patch, reject = self.patch_creator(img, item_name, patch_id=patch_id)
                annotations_patch, reject = self.patch_creator(annotations, item_name, patch_id=patch_id)
                dico_values_patches[original_index] = img_patch
                dico_values_patch_annotations[original_index] = annotations_patch
                dico_values_patch_rejects[original_index] = reject
        patches = np.stack([np.stack((dico_values_patches[k],)*3,axis=0) for k in sorted(dico_values_patches.keys())],axis=0)
        annotations = np.stack([self.make_classification_label(dico_values_patch_annotations[k]) for k in sorted(dico_values_patch_annotations.keys())],axis=0)
        rejects = np.array([dico_values_patch_rejects[k] for k in sorted(dico_values_patch_rejects.keys())])
        return patches,annotations,rejects