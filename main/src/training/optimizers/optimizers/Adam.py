from main.src.param_savers.BaseClass import BaseClass
import torch.optim as optim

from main.src.training.optimizers.optimizers.AbstractOptimizer import AbstractOptimizer


class Adam(BaseClass,AbstractOptimizer):
    def __init__(self,model,**kwargs):
        super(Adam, self).__init__(kwargs)
        self.pytorch_optimizer = optim.Adam(model.model.parameters(), **kwargs)
    @property
    def optimizer(self):
        return self.pytorch_optimizer
