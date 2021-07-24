from abc import ABC,abstractmethod

from typing import List, Tuple, Any


class AbstractGridMaker(ABC):
    """BaseClass to build a grid maker allowing to build a grid (list of coordinates) and get the coordinates of a patch"""
    def __init__(self,patch_size_final_resize: int):
        self.attr_patch_size_final_resize = patch_size_final_resize

    @abstractmethod
    def get_grid(self, img_shape: Tuple[int,...]) -> List[Tuple[int, int]]:
        """Allows to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """
    @abstractmethod
    def get_patch_transformation_matrix(self,coord_patch: Tuple[int,int]):
        """Get the transformation matrix corresponding to the coordinates provided

        Args:
            coord_patch: Tuple[int,int], x,y coordinates of the patch (as in numpy format)

        Returns:
            the np.ndarray of the transformation matrix to use to get the patch at the upper left corner of the initial image
        """
        pass
