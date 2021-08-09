from abc import ABC, abstractmethod

class AbstractOpener(ABC):
    def __init__(self,path:str):
        self.dataset = None
        self.attr_path = path
    def __getitem__(self, item):
        return self.dataset[item]