import numpy as np
import torch

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss


class MSE(BaseClass, AbstractLoss):
    def torch_compute(self, true_batch, pred_batch):
        return torch.nn.MSELoss()

    def npy_compute(self, true_batch, pred_batch):
        return np.mean(np.power(pred_batch - true_batch, 2))
