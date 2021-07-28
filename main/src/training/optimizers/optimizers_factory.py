"""Class managing the optimizers"""
import torch.optim as optim

from main.src.models.ModelFactory import ModelFactory
from main.src.enums import EnumOptimizer
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.optimizers.optimizers.AbstractOptimizer import AbstractOptimizer
from main.src.training.optimizers.optimizers.Adam import Adam
from main.src.training.optimizers.optimizers.SGD import SGD


class OptimizersFactory(BaseClass):
    """Class managing the optimizers

    Args:
        model: attr_model to optimize
        name: str EnumOptimizer, optimizer to use.

        params: other parameters for the optimizer
    """

    def create(self, model: ModelFactory, name: EnumOptimizer = EnumOptimizer.Adam, **params) -> AbstractOptimizer:
        if name == EnumOptimizer.Adam:
            return Adam(model,**params)
        elif name == EnumOptimizer.SGD:
            return SGD(model,**params)
        else:
            raise NotImplementedError(f"{name} has not been implemented")