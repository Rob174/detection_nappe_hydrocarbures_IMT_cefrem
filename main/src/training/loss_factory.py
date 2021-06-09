import torch

class LossFactory:
    def __init__(self, usage_type, preference=None):
        if preference is not None:
            preference = preference.lower()
        self.attr_loss = preference
        if usage_type == "classification":
            if preference is None or preference == "crossentropy":
                self.attr_loss = "crossentropy"
                self.loss = lambda pred,target:(-(pred+1e-7).log() * target).sum(dim=1).mean()
            elif preference == "mse":
                self.loss = torch.nn.MSELoss()
            else:
                raise NotImplementedError(f"{preference} has not been implemented")

        elif usage_type == "segmentation":
            raise NotImplementedError()
        else:
            raise NotImplementedError(f"{usage_type} has not been implemented")
    def __call__(self):
        return self.loss