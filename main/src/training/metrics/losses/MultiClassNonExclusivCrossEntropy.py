import numpy as np
import torch

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss


class MultiClassNonExclusivCrossEntropy(BaseClass, AbstractLoss):
    def __init__(self,optimizer):
        super(MultiClassNonExclusivCrossEntropy, self).__init__(optimizer)
        self.attr_name = self.__class__.__name__
    def torch_compute(self, true_batch, pred_batch):
        return torch.mean(torch.sum(-torch.log(pred_batch + 1e-7) * true_batch, dim=1))

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(np.sum(-np.log(pred_batch + 1e-7) * true_batch, axis=1))
