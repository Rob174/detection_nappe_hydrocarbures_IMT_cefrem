"""Abstract class rerpesenting future fabric constituting combinations of datasets
    (images, annotations, additionnal informations)"""

from abc import ABC,abstractmethod

from typing import Tuple, Dict

from main.src.data.Datasets.AbstractDataset import AbstractDataset


class AbstractFabricDatasets(ABC):
    """Abstract class rerpesenting future fabric constituting combinations of datasets
    (images, annotations, additionnal informations)"""
    @abstractmethod
    def __call__(self) -> Tuple[AbstractDataset,AbstractDataset,Dict]:
        """Creates the objects representing this dataset

        Must:
        - Create an image dataset object manager
        - Create an annotation dataset object manager
        - Open the image information json file
        - Provide the name corresponding to the uint values encoding the 2d array segmentation
        """
        pass