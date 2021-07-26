"""Abstract class to specify the expected interface for a callback (to be developped in the future)"""

from abc import ABC,abstractmethod


class AbstractCallback(ABC):
    """Abstract class to specify the expected interface for a callback (to be developped in the future)"""
    def __init__(self,*args,**kwargs):
        pass
    @abstractmethod
    def on_train_start(self,prediction_batch,true_batch):
        """Called for all metrics calculations """
    @abstractmethod
    def on_valid_start(self,prediction_batch,true_batch):
        """Called for all metrics calculations """

    @abstractmethod
    def on_valid_batch_end(self,prediction_batch,true_batch):
        """Called after a valid batch has been tested on the model and metrics has been calculated"""
        pass

    @abstractmethod
    def on_epoch_end(self,prediction_batch,true_batch):
        """called when an epoch edns"""
        pass