"""Class to dynamically change the interval between two epochs to complixify the training following a affine function"""
from typing import List, Optional

from main.src.data.PatchAdder.AbstractClassAdder import AbstractClassAdder
from main.src.param_savers.BaseClass import BaseClass


class PatchAdderCallback(BaseClass):
    """Class to dynamically change the interval between two epochs to complixify the training

    Args:
        step_per_epoch: int, the reduction of interval per epoch
        init_interval: int, initial interval
        class_adders: Optional[List[AbstractClassAdder]], PatchAdders to influence
    """

    def __init__(self, step_per_epoch: int = 1, init_interval: int = 1,
                 class_adders: Optional[List[AbstractClassAdder]] = None):
        """

        """
        if class_adders is None:
            class_adders = []
        self.class_adder = class_adders
        self.attr_step_per_epoch = step_per_epoch
        self.attr_init_interval = init_interval
        self.num_epoch = 0

    def on_epoch_start(self):
        """Callback called on the start of each epoch.
        Changes the interval of each of the AbstractClassAdder class under its responsability following the function
         max(init_interval-step_per_epoch*num_epoch,1)"""
        interval = max(self.attr_init_interval - self.attr_step_per_epoch * self.num_epoch, 1)
        for class_adder in self.class_adder:
            class_adder.set_interval(interval)
        self.num_epoch += 1
