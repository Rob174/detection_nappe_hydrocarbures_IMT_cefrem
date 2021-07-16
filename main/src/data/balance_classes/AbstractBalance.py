"""Base class to build a balancing operation"""

from abc import ABC


class AbstractBalance(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def filter(self, classification_label):
        """method called during training to know if we have to filter this sample or not based on its classification_label"""
