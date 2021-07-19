import unittest
import json

from main.FolderInfos import FolderInfos
from main.src.analysis.analysis.metadata.confusion_matrix.backend import ConfusionMatrixBackend


class TestConfusionMatrixBackend(unittest.TestCase):
    def test_generate_str_list(self):
        FolderInfos.init(test_without_data=True)
        with open(FolderInfos.root_folder+FolderInfos.separator.join(["main","test","unit","analysis","confusion_matrix","mock_dico.json"]),"r") as fp:
            dico = json.load(fp)
        with open(FolderInfos.root_folder+FolderInfos.separator.join(["main","src","analysis","extract_data.json"]),"r") as fp:
            extract_functions_dict = json.load(fp)
        value_functions = extract_functions_dict["confusion_matrix"]["value"]
        names_functions = extract_functions_dict["confusion_matrix"]["names"]
        confusion_matrix = ConfusionMatrixBackend(dico,value_functions,names_functions)
        print(confusion_matrix.generate_str_list())