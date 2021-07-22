import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class MarginCheck(BaseClass):
    def __init__(self,threshold: int = 1000):
        self.attr_threshold = threshold

    def check_reject(patch: np.ndarray, threshold_px: int):
        if len(patch[
                   patch == 0]) > threshold_px:  # 0 ns or 1 ms (sometimes) for the condition. 1 ms and 5 ms for the len(patch.... . 0 ns for the int(.....
            return True
        return False
