import unittest

from main.src.data.TwoWayDict import TwoWayDict
from main.src.training.confusion_matrix.ConfusionMatrixCallback import ConfusionMatrixCallback


class TestConfusionMatrixCallback(unittest.TestCase):
    def test_confusion_matrix_compiles(self):
        callback = ConfusionMatrixCallback(TwoWayDict({
            0:"other",
            1:"seep",
            2:"spill",
        }))
        self.assertEqual(
            [(0, 'lambda label:label[0] == 0 and label[1] == 0 and label[2] == 0'),
            (1, 'lambda label:label[0] == 0 and label[1] == 0 and label[2] == 1'),
            (2, 'lambda label:label[0] == 0 and label[1] == 1 and label[2] == 0'),
            (3, 'lambda label:label[0] == 0 and label[1] == 1 and label[2] == 1'),
            (4, 'lambda label:label[0] == 1 and label[1] == 0 and label[2] == 0'),
            (5, 'lambda label:label[0] == 1 and label[1] == 0 and label[2] == 1'),
            (6, 'lambda label:label[0] == 1 and label[1] == 1 and label[2] == 0'),
            (7, 'lambda label:label[0] == 1 and label[1] == 1 and label[2] == 1')],
            list(callback.class_mappings.items()))
        self.assertEqual(
            [   "nothing",
                "spill",
                "seep",
                "seep_spill",
                "other",
                "other_spill",
                "other_seep",
                "other_seep_spill"],
            callback.class_names)


if __name__ == '__main__':
    unittest.main()
