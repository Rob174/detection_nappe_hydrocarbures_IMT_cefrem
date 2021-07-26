"""Base class to create label modifier"""

from abc import ABC, abstractmethod

import numpy as np

from main.src.param_savers.BaseClass import BaseClass


class AbstractLabelModifier(ABC):
    def __init__(self):
        self.initial_label = None

    @abstractmethod
    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        """Creates the Generators label based on the annotation patch image

        Args:
            annotation: np.ndarray 2d containing the class of each pixel encoded as uint8

        Returns:
            annotation_modified: the Generators label modified

        """
        pass
    def get_initial_label(self):
        assert self.initial_label is not None, "make_classification_label must be called before get_initial_label"
        return self.initial_label
