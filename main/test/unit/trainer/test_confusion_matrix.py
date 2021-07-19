import unittest

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
