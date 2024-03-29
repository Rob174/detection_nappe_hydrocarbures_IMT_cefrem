"""Creates a confusion matrix on each validation sample provided with the update_matrix """

import numpy as np

from typing import Dict, List

from main.src.param_savers.BaseClass import BaseClass


class ConfusionMatrix(BaseClass):
    def __init__(self,class_mappings: Dict[int,str],class_names: List[str]):
        """

        Args:
            class_mappings: dict of int,  str with for each name the function to extract on an individual prediction (no batch dim)
            class_names: names in the order to indicate how to map index to real names
        """
        self.attr_class_names = class_names
        self.attr_class_mappings = class_mappings
        self.num_matrix_classes = len(self.attr_class_mappings)
        self.matrix = np.zeros((self.num_matrix_classes,self.num_matrix_classes))
        self.tot_pred = np.zeros((self.num_matrix_classes,))
        self.tot_true = np.zeros((self.num_matrix_classes,))
        self.attr_full_matrix = None
    def apply_threshold(self,image: np.ndarray):
        """Function the apply a 0.5 threshold to the label"""
        new_image = np.copy(image)
        new_image[image > 0.5] = 1.
        new_image[image < 0.5] = 0.
        return new_image
    def update_matrix(self,prediction,true):
        """Creates the confusion matrix corresponding to the validation batch

        Args:
            prediction: np array of shape [batch_size,num_classes]
            true: np array of shape [batch_size,num_classes]

        Returns:

        """
        prediction = self.apply_threshold(prediction)
        self.matrix = np.zeros((self.num_matrix_classes,self.num_matrix_classes))
        self.tot_pred = np.zeros((self.num_matrix_classes,))
        self.tot_true = np.zeros((self.num_matrix_classes,))
        for pred_value,true_value in zip(prediction,true):
            class_matrix_pred = "none"
            class_matrix_true = "none"
            for class_index,function_test in self.attr_class_mappings.items():
                if eval(function_test)(pred_value):
                    class_matrix_pred = class_index
                if eval(function_test)(true_value):
                    class_matrix_true = class_index
            if class_matrix_true == "none" or class_matrix_pred == "none":
                raise Exception(f"One class at least has no index found with \nclass_matrix_true={class_matrix_true} and class_matrix_pred={class_matrix_pred}\n")
            self.matrix[class_matrix_pred,class_matrix_true] += 1
            self.tot_pred[class_matrix_pred] += 1
            self.tot_true[class_matrix_true] += 1

        full_matrix = np.zeros((self.num_matrix_classes+1,self.num_matrix_classes+1))
        full_matrix[:self.num_matrix_classes,:self.num_matrix_classes] = self.matrix
        full_matrix[:self.num_matrix_classes, -1] = self.tot_pred
        full_matrix[-1, :self.num_matrix_classes] = self.tot_true
        full_matrix[-1,-1] = np.sum(np.diag(self.matrix))

        self.attr_full_matrix = full_matrix.tolist()