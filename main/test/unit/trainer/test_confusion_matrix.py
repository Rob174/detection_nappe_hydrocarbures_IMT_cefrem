import unittest
from functools import reduce

from main.src.training.confusion_matrix.ConfusionMatrix import ConfusionMatrix

import numpy as np


class test_confusion_matrix(unittest.TestCase):
    def test_confusion_matrix_compiles(self):
        confusion_matrix = ConfusionMatrix(class_mappings={
            0:"lambda label:label[0] == 0 and label[1] == 0",
            1:"lambda label:label[0] == 0 and label[1] == 1",
            2:"lambda label:label[0] == 1 and label[1] == 0",
            3:"lambda label:label[0] == 1 and label[1] == 1"
        },
            class_names=[
                "Nothing",
                "Spill",
                "Seep",
                "SeepSpill"
            ]
        )
        true = np.random.randint(0,2,(100,2))
        pred = np.random.randint(0, 2, (100, 2))
        confusion_matrix.update_matrix(pred,true)
        print("\n".join(list(map(lambda x:"\t".join(x),confusion_matrix.attr_full_matrix))))
    def test_confusion_matrix_diag(self):
        prediction = np.array([[1,0.],[0.,1],[0.,0.],[1.,0.],[1,0.],
                               [0.,1],[1,0.],[0.,1],[1,0.],[0.,1]])
        true = np.array([[1,0.],[0.,1],[0.,0.],[1.,0.],[1,0.],
                         [1.,1],[0,0.],[1.,1],[0,0.],[1.,1]])

        prediction = prediction[:10]
        true = true[:10]
        confusion_matrix = ConfusionMatrix(class_mappings={
            0:"lambda label:label[0] == 0 and label[1] == 0",
            1:"lambda label:label[0] == 0 and label[1] == 1",
            2:"lambda label:label[0] == 1 and label[1] == 0",
            3:"lambda label:label[0] == 1 and label[1] == 1"
        },
            class_names=[
                "Nothing",
                "Spill",
                "Seep",
                "SeepSpill"
            ]
        )

        def npy_compute(true_batch, pred_batch):
            new_pred = np.copy(pred_batch)
            new_true = np.copy(true_batch)
            new_pred[pred_batch > 0.5] = 1.
            new_pred[pred_batch <= 0.5] = 0.
            new_true[true_batch > 0.5] = 1.
            new_true[true_batch <= 0.5] = 0.
            return np.sum(np.abs(new_pred - new_true)) # nb errors seep or spill
        confusion_matrix.update_matrix(prediction,true)
        # matrice de confusion a plus d'éléments sur la diagonale qu'en réalité
        diff = np.sum(reduce(lambda x,y:x*y,list(true.shape)))-npy_compute(true,prediction)
        matrice_diag = np.sum(np.diag(confusion_matrix.matrix))
        self.assertEqual(matrice_diag,diff)
