"""
Module that gather classes creating and managing patches

Each of the class must have the following structure:


>>> class MyPatch_creator(BaseClass):
...     def __init__(self, *args, **kargs):
...         self.attr_name = self.__class__.__name__
...         self.attr_global_name = "patch_creator"
...     def num_available_patches(self,image: np.ndarray ) -> int:
...         # count the number of patches available for an original image
...     def __call__(self, image: np.ndarray,image_name: str, patch_id: int,count_reso=False):
...         return self.call(image,image_name,patch_id,count_reso)
...     def call(self, image: np.ndarray,image_name: str, patch_id: int,*args,**kargs) -> Tuple[np.ndarray,bool]:
...         # Creates the patch with provided data and indicate if it is rejected
...
...     def get_position_patch(self,patch_id: int, input_shape):
...         # Retrieve the pixel coordinates of the upper left corner
"""