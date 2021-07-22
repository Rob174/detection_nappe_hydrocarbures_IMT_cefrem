from enum import Enum
from typing import Optional, List, Dict

import torch
import numpy as np

from main.src.enums import *
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager
from main.src.training.optimizers_factory import OptimizersFactory


class LossFactory(BaseClass, AbstractMetricManager):
    """Class managing possible losses

    Args:
        usage_type: EnumUsage enum, indicate for which purpose we want a loss.

        preference: optional EnumLoss, the loss to use. MulticlassnonExlusivCrossentropy by default

    """

    def __init__(self, optimizer: OptimizersFactory, usage_type: EnumUsage, preference: Optional[EnumLoss] = None):
        super(LossFactory, self).__init__()
        self.optimizer: OptimizersFactory = optimizer
        self.attr_loss = preference
        if usage_type == EnumUsage.Classification:
            if preference is None or preference == EnumLoss.MulticlassnonExlusivCrossentropy:
                self.attr_loss = EnumLoss.MulticlassnonExlusivCrossentropy
                self.loss = lambda pred, target: torch.mean(torch.sum(-torch.log(pred + 1e-7) * target, dim=1))
                self.npy_loss = lambda pred, target: np.mean(np.sum(-np.log(pred + 1e-7) * target, axis=1))
            elif preference == EnumLoss.BinaryCrossentropy:
                self.loss = torch.nn.BCELoss()
                self.npy_loss = lambda pred, target: np.mean(-(target*np.log(np.clip(pred,np.exp(-100),None)) +
                                                                       (1-target)*np.log(np.clip(1-pred,np.exp(-100),None)))) # Checked with debugger
            elif preference == EnumLoss.MSE:
                self.loss = torch.nn.MSELoss()
                self.npy_loss = lambda pred, target: np.mean(np.power(pred-target,2))
            else:
                raise NotImplementedError(f"{preference} has not been implemented")

        elif usage_type == EnumUsage.Segmentation:
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
        self.attr_global_name = "loss"
        self.attr_loss_values: Dict[str, List[float]] = {EnumDataset.Train: [], EnumDataset.Valid: []}

    def call(self, prediction_gpu, output_gpu, prediction_npy,output_npy, dataset: EnumDataset = EnumDataset.Train) -> float:
        loss = self.loss(prediction_gpu, output_gpu)
        if dataset == EnumDataset.Train:
            loss.backward()
            self.optimizer().step()
        current_loss = self.npy_loss(prediction_npy,output_npy)
        self.attr_loss_values[dataset].append(current_loss)
        return current_loss

    def __call__(self, prediction_gpu, output_gpu, prediction_npy,output_npy, dataset: EnumDataset = EnumDataset.Train) -> float:
        return self.call(prediction_gpu, output_gpu,prediction_npy,output_npy, dataset)

    def get_last_metric(self, name: Enum) -> float:
        return self.attr_loss_values[EnumDataset.Valid][-1]
