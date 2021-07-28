import torch.optim as optim

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.optimizers.optimizers.AbstractOptimizer import AbstractOptimizer


class SGD(BaseClass, AbstractOptimizer):
    def __init__(self, model, **kwargs):
        super(SGD, self).__init__(kwargs)
        self.pytorch_optimizer = optim.SGD(model.model.parameters(), **kwargs)

    @property
    def optimizer(self):
        return self.pytorch_optimizer
