"""Abstract class to specify the expected interface for a callback (to be developped in the future)"""

from abc import ABC,abstractmethod


class AbstractCallback(ABC):
    """Abstract class to specify the expected interface for a callback (to be developped in the future)"""
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def on_valid_batch_end(self,prediction_batch,true_batch):
        """Called after a valid batch has been tested on the model"""
        pass
