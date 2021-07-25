"""Class and function to access training and valid samples in the dataset factory"""
from typing import List


class TrValidSplit:
    """Wrapper to return the correct iterator for the training or validation dataset"""
    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __iter__(self):
        return self.dataset.__iter__(dataset=self.name)

    def len(self):
        return self.dataset.__len__(dataset=self.name)


def trvalidsplit(dataset) -> List[TrValidSplit]:
    """Function to create two objects returning the correct iterator to get the training and validation dataset"""
    return [TrValidSplit(dataset, name="tr"), TrValidSplit(dataset, name="valid")]
