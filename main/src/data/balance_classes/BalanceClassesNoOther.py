"""Balance classes by excluding patches where there is only the other class"""

import numpy as np

from main.src.data.balance_classes.AbstractBalance import AbstractBalance
from main.src.param_savers.BaseClass import BaseClass


class BalanceClassesNoOther(BaseClass, AbstractBalance):
    def __init__(self, other_index):
        """Balance classes by excluding patches where there is only the other class

        Args:
            other_index: index of the class other
        """
        super().__init__()
        self.attr_other_index = other_index
        self.attr_num_accepted = 0
        self.attr_name = self.__class__.__name__  # save the name of the class used for reproductibility purposes
        self.attr_global_name = "balance"  # save a more compehensible name

    def filter(self, classification_label):
        """method called during training to know if we have to filter this sample or not based on its classification_label

        Args:
            classification_label:  label with ones of a class is on the image and 0 if not. !! Must be provided by NoLabelModifier make_classification_label method for the shape of the labels (with full details)

        Returns:
            bool, tell if the sample is accepted or rejected

        """
        if len(classification_label[classification_label > 0]) == 1 and np.argmax(
                classification_label) == self.attr_other_index:
            return True
        self.attr_num_accepted += 1
        return False
