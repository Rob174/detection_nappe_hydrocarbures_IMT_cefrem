import numpy as np
from typing import List, Tuple


class GridMaker:
    def __init__(self,patch_size_final_resize: int):
        self.attr_patch_size_final_resize = patch_size_final_resize
    def get_grid(self, img_shape, partial_transformation_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Allow to create the adapted grid to the transformation as resize and rotation are involved in the process.


        Args:
            img_shape: shape of the original image
            partial_transformation_matrix: transformation matrix include all augmentations (return values of choose_new_augmentation)

        Returns:
            iterator that produces tuples with coordinates of each upper left corner of each patch
        """
        rows, cols = img_shape[:2]
        original_mapped_corner1 = partial_transformation_matrix.dot([cols, rows, 1])
        original_mapped_corner2 = partial_transformation_matrix.dot([cols, rows, 1])
        max_rows = max(original_mapped_corner1[1], original_mapped_corner2[1]) - self.attr_patch_size_final_resize
        max_cols = max(original_mapped_corner1[0], original_mapped_corner2[0]) - self.attr_patch_size_final_resize
        cols_coords = np.arange(0, max_cols, self.attr_patch_size_final_resize)
        rows_coords = np.arange(0, max_rows, self.attr_patch_size_final_resize)
        coords = list(zip(*list(x.flat for x in np.meshgrid(rows_coords, cols_coords))))
        return coords
    def get_patch_transformation_matrix(self,coord_patch: Tuple[int,int]):
        return np.array([[1, 0, -coord_patch[1]],
                         [0, 1, -coord_patch[0]],
                         [0, 0, 1]])