import torch

from main.src.param_savers.BaseClass import BaseClass


class LossFactory(BaseClass):
    """Class managing possible losses

    Args:
        usage_type: str enum, indicate for which purpose we want a loss. Currently supported:
        - "classification"
        - "segmentation"

        preference: optional str enum, indicate the preference for loss. Currently supported:
        - "multiclassnonexlusivcrossentropy"
        - "binarycrossentropy"
        - "mse"

    """
    def __init__(self, usage_type, preference=None):
        if preference is not None:
            preference = preference.lower()
        self.attr_loss = preference
        if usage_type == "classification":
            if preference is None or preference == "multiclassnonexlusivcrossentropy":
                self.attr_loss = "multiclassnonexlusivcrossentropy"
                self.loss = lambda pred,target:torch.mean(torch.sum(-torch.log(pred+1e-7) * target,dim=1))
            elif preference == "binarycrossentropy":
                self.attr_loss = "binarycrossentropy"
                self.loss = torch.nn.BCELoss()
            elif preference == "mse":
                self.attr_loss = "mse"
                self.loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"{preference} has not been implemented")

        elif usage_type == "segmentation":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
        self.attr_global_name = "loss"
    def call(self):
        return self.loss
    def __call__(self):
        return self.call()