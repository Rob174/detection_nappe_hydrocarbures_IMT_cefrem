from typing import List, Optional

from main.src.data.PatchAdder.AbstractClassAdder import AbstractClassAdder
from main.src.param_savers.BaseClass import BaseClass


class PatchAdderCallback(BaseClass):
    def __init__(self, step_per_epoch: int = 1, init_interval: int = 1, class_adders: Optional[List[AbstractClassAdder]]=None):
        if class_adders is None:
            class_adders = []
        self.class_adder = class_adders
        self.attr_step_per_epoch = step_per_epoch
        self.attr_init_interval = init_interval
        self.num_epoch = 0
    def on_epoch_start(self):
        interval = max(self.attr_init_interval-self.attr_step_per_epoch*self.num_epoch,1)
        for class_adder in self.class_adder:
            class_adder.set_interval(interval)
        self.num_epoch += 1
