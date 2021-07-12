import json, random

import numpy as np

from h5py import File
from typing import Optional, Tuple

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass


class NoClassPatchAdder(BaseClass):
    """Object to provide similar interface as OtherClassPatchAdder"""
    def __init__(self,*args,**kwargs):
        pass


    def generate_if_required(self) -> Optional[Tuple[np.ndarray,np.ndarray,np.ndarray,str]]:
        return None


