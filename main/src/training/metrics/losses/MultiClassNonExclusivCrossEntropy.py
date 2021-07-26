from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
import torch
import numpy as np


class MultiClassNonExclusivCrossEntropy(BaseClass,AbstractLoss):
    def torch_compute(self,true_batch,pred_batch):
        return torch.mean(torch.sum(-torch.log(pred_batch + 1e-7) * true_batch, dim=1))
    def npy_compute(self,true_batch,pred_batch):
        return np.mean(np.sum(-np.log(pred_batch + 1e-7) * true_batch, axis=1))
