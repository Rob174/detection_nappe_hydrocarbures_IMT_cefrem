import numpy as np

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory


def moving_mean(x,window):
    new_values = []
    for i in range(len(x)-window):
        new_values.append(np.mean(x[i:min(i+window,len(x))]))
    return new_values

class RGB_Overlay_Patch:
    def __init__(self,dataset_name="sentinel1",usage_type="classification",patch_creator="fixed_px",grid_size=1000,input_size=256):
        FolderInfos.init(test_without_data=False)
        self.dataset = DatasetFactory(dataset_name=dataset_name, usage_type=usage_type, patch_creator=patch_creator,grid_size=grid_size , input_size=input_size)
    def __call__(self, name_img,blending_factor=0.5):
        original_img = self.dataset.attr_dataset.images[name_img]
        overlay = np.zeros((*original_img.shape[:-1],3))
        normalize = lambda x:(x-np.min(original_img))/(np.max(original_img)-np.min(original_img))
        for id,[input,output] in enumerate(self.dataset.attr_dataset.all_patches_of_image(name_img)):
            pos_x,pos_y = self.dataset.attr_patch_creator.get_position_patch(id)
            if len(output) > 3:
                raise Exception(f"{len(output)} classes : Too much : cannot use this vizualization technic")
            input = normalize(input)
            input = np.stack((input,)*3) # convert in rgb
            color = np.ones((self.dataset.attr_patch_creator.attr_grid_size_px,
                                  self.dataset.attr_patch_creator.attr_grid_size_px,3))
            for i,c in enumerate(output):
                color[:,:,i] *= c
            overlay[pos_x:pos_x+self.dataset.attr_patch_creator.attr_grid_size_px,
                    pos_y:pos_y+self.dataset.attr_patch_creator.attr_grid_size_px,
                    :] = input * (1-blending_factor) + color * blending_factor
        return overlay