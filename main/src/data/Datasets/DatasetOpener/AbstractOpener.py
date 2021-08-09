from abc import ABC, abstractmethod

class AbstractOpener(ABC):
    def __init__(self,path:str):
        self.dataset = None
        self.attr_path = path
