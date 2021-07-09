from enum import Enum
from typing import Optional, List, Dict

import torch

from main.src.data.enums import EnumUsage
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.AbstractMetricManager import AbstractMetricManager
from main.src.training.enums import EnumLoss, EnumDataset
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
            elif preference == EnumLoss.BinaryCrossentropy:
                self.loss = torch.nn.BCELoss()
            elif preference == EnumLoss.MSE:
                self.loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"{preference} has not been implemented")

        elif usage_type == EnumUsage.Segmentation:
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
        self.attr_global_name = "loss"
        self.attr_loss_values: Dict[str, List[float]] = {EnumDataset.Train: [], EnumDataset.Valid: []}

    def call(self, prediction_gpu, output_gpu, dataset: EnumDataset = EnumDataset.Train) -> float:
        loss = self.loss(prediction_gpu, output_gpu)
        loss.backward()
        self.optimizer().step()
        current_loss = loss.item()
        self.attr_loss_values[dataset].append(current_loss)
        return current_loss

    def __call__(self, prediction_gpu, output_gpu, dataset: EnumDataset = EnumDataset.Train) -> float:
        return self.call(prediction_gpu, output_gpu, dataset)

    def get_last_metric(self, name: Enum) -> float:
        return self.attr_loss_values[EnumDataset.Valid][-1]
