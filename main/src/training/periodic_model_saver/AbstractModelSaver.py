from abc import ABC,abstractmethod

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.metrics.metrics_factory import MetricsFactory


class AbstractModelSaver(BaseClass,ABC):
    def __init__(self,loss: LossFactory,metrics: MetricsFactory):
        pass
    @abstractmethod
    def save_model_if_required(self,model,epoch,iteration):
        pass