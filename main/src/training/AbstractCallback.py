from abc import ABC,abstractmethod


class AbstractCallback(ABC):
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def on_valid_batch_end(self,prediction_batch,true_batch):
        pass
