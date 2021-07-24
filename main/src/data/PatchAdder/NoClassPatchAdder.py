"""Class to provide similar interface as OtherClassPatchAdder"""

from typing import Optional, Tuple

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class NoClassPatchAdder(BaseClass):
    """Object to provide similar interface as OtherClassPatchAdder"""

    def __init__(self,interval: int, *args, **kwargs):
        pass

    def generate_if_required(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        """Generates no patchs"""
        return None
