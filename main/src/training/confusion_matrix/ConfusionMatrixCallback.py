from main.src.data.TwoWayDict import TwoWayDict, Way
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.confusion_matrix.ConfusionMatrix import ConfusionMatrix

from itertools import combinations,product


class ConfusionMatrixCallback(BaseClass,AbstractCallback):
    def __init__(self,dico_class_mappings: TwoWayDict):
        super(ConfusionMatrixCallback, self).__init__()
        class_mappings = {}
        class_names = []
        i = 0
        for id_names in combinations(dico_class_mappings.items(dico_chosen=Way.ORIGINAL_WAY),len(dico_class_mappings)):
            for presence in product([0,1],repeat=len(dico_class_mappings)):
                    class_mappings[i] = "lambda label:"
                    final_name = []
                    l= []
                    for (id,name),present in zip(id_names,presence):
                        l.append(f"label[{id}] == {present}")
                        if present == 1:
                            if isinstance(name,list) or isinstance(name,tuple): # TODO cuppling to solve
                                name = name[0]
                            final_name.append(name)
                    if len(final_name) == 0:
                        final_name = ["nothing"]
                    class_mappings[i] += " and ".join(l)
                    class_names.append("_".join(final_name))
                    i += 1
        self.class_mappings = class_mappings
        self.class_names = class_names
        self.attr_confusion_matrix = ConfusionMatrix(class_mappings,class_names)
    def on_valid_batch_end(self,prediction_batch,true_batch):
        self.attr_confusion_matrix.update_matrix(prediction_batch,true_batch)