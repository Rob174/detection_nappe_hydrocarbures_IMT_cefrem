import numpy as np
import torch

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss


class BinaryCrossEntropy(BaseClass, AbstractLoss):
    def __init__(self,optimizer):
        super(BinaryCrossEntropy, self).__init__(optimizer)
        self.attr_name = self.__class__.__name__
        self.loss = torch.nn.BCELoss()
    def torch_compute(self, true_batch, pred_batch):
        return self.loss(pred_batch,true_batch)

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(-(true_batch * np.log(np.clip(pred_batch, np.exp(-100), None)) + (1 - true_batch) * np.log(
            np.clip(1 - pred_batch, np.exp(-100), None))))
