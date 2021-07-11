from main.src.analysis.analysis.metadata.MetadataManager import MetadataManager
from main.src.analysis.analysis.metadata.Formatters.NoFormatter import NoFormatter
from main.src.analysis.analysis.metrics.LossManager import LossManager


class MetricManager:
    def __init__(self,dico_metrics,dico_data):
        self.dico_metrics = dico_metrics
        self.dico_data = dico_data
        self.metadata_manager = MetadataManager(self.dico_metrics,self.dico_data,NoFormatter())
        self.loss_manager = LossManager()
    def extract_from_list_of_possibilities(self,functions_list,dico_data):
        return self.metadata_manager.extract_from_list_of_possibilities(functions_list,dico_data)
    def extract_from_uniq_function(self,function,dico_data):
        return self.metadata_manager.extract_from_uniq_function(function,dico_data)
    def get(self,metric_name):
        data = None
        if isinstance(self.dico_metrics[metric_name], list):
            data = self.extract_from_list_of_possibilities(self.dico_metrics[metric_name], self.dico_data)
        elif isinstance(self.dico_metrics[metric_name], str):
            data = self.extract_from_uniq_function(self.dico_metrics[metric_name], self.dico_data)
        else:
            raise NotImplementedError
        if data is None:
            data = {"tr_values":[],"valid_values":[]}
        if metric_name == "loss":
            data = self.loss_manager.reformat(data)
        return data