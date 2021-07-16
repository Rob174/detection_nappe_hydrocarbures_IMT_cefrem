"""Abstract class to create new class adder"""

from abc import ABC
from typing import Optional, Tuple

import numpy as np


class AbstractClassAdder(ABC):
    def generate_if_required(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        return None
