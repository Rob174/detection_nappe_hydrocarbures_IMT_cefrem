from typing import Union

from main.FolderInfos import FolderInfos
from main.src.data.preprocessing.open_files import open_raster
import numpy as np


class Test_images:
    def __init__(self):
        self.names = ["20190601_S1A","20190601_S1B","20190618"]
        self.current_name = ""
    def get_rasters(self,selector: Union[str,int]) -> np.ndarray:
        name: str = ""
        if isinstance(selector,str):
            name = list(filter(lambda x:x==selector,self.names))[0]
        elif isinstance(selector,int):
            name = self.names[selector]
        else:
            raise Exception(f"Invalid selector: {selector}")
        self.current_name = name
        return open_raster(FolderInfos.data_test+"Sigma0_VV_db_"+name+".img")