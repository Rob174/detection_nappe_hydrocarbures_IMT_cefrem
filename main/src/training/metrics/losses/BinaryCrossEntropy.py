from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
import torch
import numpy as np


class BinaryCrossEntropy(BaseClass,AbstractLoss):
    def torch_compute(self,true_batch,pred_batch):
        return torch.nn.BCELoss()
    def npy_compute(self,true_batch,pred_batch):
        return np.mean(-(true_batch*np.log(np.clip(pred_batch,np.exp(-100),None)) + (1-true_batch)*np.log(np.clip(1-pred_batch,np.exp(-100),None))))
