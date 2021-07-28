from abc import ABC


class AbstractOptimizer(ABC):
    def __init__(self,**kwargs):
        self.attr_arguments = kwargs
    @property
    def optimizer(self):
        raise NotImplementedError
    def zero_grad(self):
        """Put the gradient to 0"""
        self.optimizer.zero_grad()
    def step(self):
        self.optimizer.step()