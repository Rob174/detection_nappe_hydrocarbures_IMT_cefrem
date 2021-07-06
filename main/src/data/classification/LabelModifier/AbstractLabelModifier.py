from abc import ABC,abstractmethod


class AbstractLabelModifier(ABC):
    def __init__(self):
        pass

    def make_classification_label(self, annotation: np.ndarray) -> np.ndarray:
        """Creates the classification label based on the annotation patch image

        Args:
            annotation: np.ndarray 1d containing the probability that the patch contain the classes as specified in NoLabelModifier make_classification_label method

        Returns:
            annotation_modified: the classification label modified

        """
        pass