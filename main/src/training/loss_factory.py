import torch

from main.src.param_savers.BaseClass import BaseClass


class LossFactory(BaseClass):
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
                self.loss = torch.nn.CrossEntropyLoss()
            elif preference == "mse":
                self.attr_loss = "mse"
                self.loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"{preference} has not been implemented")

        elif usage_type == "segmentation":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
    def __call__(self):
        return self.loss