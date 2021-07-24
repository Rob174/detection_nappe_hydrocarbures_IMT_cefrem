"""BaseClass to build an augmentation applier"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

import numpy as np

from main.src.data.GridMaker.GridMaker import GridMaker


class AbstractApplier(ABC):
    """BaseClass to build an augmentation applier"""

    def __init__(self, grid_maker: GridMaker, patch_size_final_resize: int, *args, **kwargs):
        self.grid_maker = grid_maker
        self.patch_size_final_resize = patch_size_final_resize

    @abstractmethod
    def transform(self, data: Any, partial_transformation_matrix: np.ndarray,
                  patch_upper_left_corner_coords: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the transformation on the input data (can be of various types) and extract the correct patch thanks to the grid maker"""
        pass
