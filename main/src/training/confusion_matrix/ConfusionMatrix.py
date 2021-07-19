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
    def update_matrix(self,prediction,true):
        """Creates the confusion matrix corresponding to the validation batch

        Args:
            prediction: np array of shape [batch_size,num_classes]
            true: np array of shape [batch_size,num_classes]

        Returns:

        """
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

        # tot = np.sum(self.matrix)
        # full_matrix_percent = np.copy(full_matrix) / tot * 100
        #
        # final_matrix = np.empty((self.num_matrix_classes+1,self.num_matrix_classes+1)).tolist()
        # for x in range(self.num_matrix_classes+1):
        #     for y in range(self.num_matrix_classes+1):
        #         final_matrix[x][y] = f"{full_matrix[x,y]}<br>{full_matrix_percent[x,y]:.2f}%"
        # final_matrix.insert(0,[]+self.attr_class_names)
        # for l_id in range(self.num_matrix_classes):
        #     final_matrix[l_id+1].insert(0,self.attr_class_names[l_id])
        self.attr_full_matrix = full_matrix.tolist()