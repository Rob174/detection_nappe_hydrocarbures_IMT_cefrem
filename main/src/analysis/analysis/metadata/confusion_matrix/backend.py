import numpy as np

class ConfusionMatrixBackend:
    def __init__(self,dico,access_functions,access_names):
        self.matrix = None
        it = iter(access_functions)
        while self.matrix is None:
            try:
                self.matrix = np.array(eval(next(it))(dico))
            except KeyError:
                pass
            except StopIteration:
                raise Exception(f"Matrix not found with functions {access_functions}")
        self.names = None
        it = iter(access_names)
        while self.names is None:
            try:
                self.names = eval(next(it))(dico)
            except KeyError:
                pass
            except StopIteration:
                raise Exception(f"Matrix not found with functions {access_names}")
    def generate_str_list(self):
        tot = np.sum(self.matrix[:-1,:-1])
        full_matrix_percent = np.copy(self.matrix) / tot * 100
        num_matrix_classes = len(self.matrix[:-1,:-1])
        final_matrix = np.empty((num_matrix_classes + 1, num_matrix_classes + 1)).tolist()
        for x in range(num_matrix_classes + 1):
            for y in range(num_matrix_classes + 1):
                final_matrix[x][y] = f"{self.matrix[x, y]}<br>{full_matrix_percent[x, y]:.2f}%"
        return final_matrix
