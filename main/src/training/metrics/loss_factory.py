"""Class managing possible losses to objects"""
from typing import Optional, List, Dict

import torch
import numpy as np

from main.src.enums import *
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
from main.src.training.metrics.losses.BinaryCrossEntropy import BinaryCrossEntropy
from main.src.training.metrics.losses.MSE import MSE
from main.src.training.metrics.losses.MultiClassNonExclusivCrossEntropy import MultiClassNonExclusivCrossEntropy
from main.src.training.optimizers_factory import OptimizersFactory


class LossFactory(BaseClass):
    """Class managing possible losses to objects"""
    @staticmethod
    def create(optimizer: OptimizersFactory, usage_type: EnumUsage, preference: Optional[EnumLoss] = None) -> AbstractLoss:
        """Create the loss required by the user
        Args:
            optimizer: optimizer to use
            usage_type: EnumUsage enum, indicate for which purpose we want a loss.
            preference: optional EnumLoss, the loss to use. MulticlassnonExlusivCrossentropy by default

        Returns:

        """
        losses = {
            EnumUsage.Classification: {
                EnumLoss.MulticlassnonExlusivCrossentropy: MultiClassNonExclusivCrossEntropy(optimizer),
                EnumLoss.BinaryCrossentropy: BinaryCrossEntropy(optimizer),
                EnumLoss.MSE: MSE(optimizer)
            },
            EnumUsage.Segmentation:{}
        }
        if usage_type == EnumUsage.Classification:
            if preference is None:
                loss = losses[EnumUsage.Classification][EnumLoss.MulticlassnonExlusivCrossentropy]
            else:
                loss = losses[EnumUsage.Classification][preference]

        elif usage_type == EnumUsage.Segmentation:
            loss = losses[EnumUsage.Segmentation][preference]
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
        return loss