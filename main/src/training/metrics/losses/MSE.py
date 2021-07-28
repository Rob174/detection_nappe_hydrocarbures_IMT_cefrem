import numpy as np
import torch

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss


class MSE(BaseClass, AbstractLoss):
    def __init__(self,optimizer):
        super(MSE, self).__init__(optimizer)
        self.attr_name = self.__class__.__name__
    def torch_compute(self, true_batch, pred_batch):
        return torch.nn.MSELoss()

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(np.power(pred_batch - true_batch, 2))
