
import torch.optim as optim

from main.src.param_savers.BaseClass import BaseClass


class OptimizersFactory(BaseClass):
    """Class managing the optimizers

    Args:
            model: model to optimize
            name: str enum, optimizer to use. Currently supported:
            - "adam"
            - "sgd"
            **params: other parameters for the optimizer
    """
    def __init__(self,model,name="adam",**params):
        self.attr_name = name
        if name == "adam":
            self.attr_params = {k:v for k,v in params.items() if k in ["lr","eps"]}
            self.optimizer = optim.Adam(model.parameters(),lr=params["lr"],eps=params["eps"])
        elif name == "sgd":
            self.attr_params = {k:v for k,v in params.items() if k in ["lr"]}
            self.optimizer = optim.SGD(model.parameters(),lr=params["lr"])
        else:
            raise NotImplementedError(f"{name} has not been implemented")
        self.attr_global_name = "optimizer"
    def call(self):
        return self.optimizer
    def __call__(self):
        return self.call()