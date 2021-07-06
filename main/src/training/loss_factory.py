from typing import Optional

import torch

from main.src.data.enums import EnumUsage
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.enums import EnumLoss


class LossFactory(BaseClass):
    """Class managing possible losses

    Args:
        usage_type: EnumUsage enum, indicate for which purpose we want a loss.

        preference: optional EnumLoss, the loss to use. MulticlassnonExlusivCrossentropy by default

    """
    def __init__(self, usage_type: EnumUsage, preference: Optional[EnumLoss]=None):

        self.attr_loss = preference
        if usage_type == EnumUsage.Classification:
            if preference is None or preference == EnumLoss.MulticlassnonExlusivCrossentropy:
                self.loss = lambda pred,target:torch.mean(torch.sum(-torch.log(pred+1e-7) * target,dim=1))
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
    def call(self):
        return self.loss
    def __call__(self):
        return self.call()