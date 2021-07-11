import torch.optim as optim

from main.src.models.ModelFactory import ModelFactory
from main.src.models.enums import EnumOptimizer
from main.src.param_savers.BaseClass import BaseClass


class OptimizersFactory(BaseClass):
    """Class managing the optimizers

    Args:
        model: attr_model to optimize
        name: str EnumOptimizer, optimizer to use.

        params: other parameters for the optimizer
    """

    def __init__(self, model: ModelFactory, name: EnumOptimizer = EnumOptimizer.Adam, **params):
        self.attr_name = name
        if name == EnumOptimizer.Adam:
            self.attr_params = {k: v for k, v in params.items() if k in ["lr", "eps"]}
            self.optimizer = optim.Adam(model.model.parameters(), lr=params["lr"], eps=params["eps"])
        elif name == EnumOptimizer.SGD:
            self.attr_params = {k: v for k, v in params.items() if k in ["lr"]}
            self.optimizer = optim.SGD(model.model.parameters(), lr=params["lr"])
        else:
            raise NotImplementedError(f"{name} has not been implemented")
        self.attr_global_name = "optimizer"

    def zero_grad(self):
        self.optimizer.zero_grad()

    def call(self):
        return self.optimizer

    def __call__(self):
        return self.call()
