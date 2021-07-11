from main.src.analysis.analysis.metadata.MetadataManager import MetadataManager
from main.src.analysis.analysis.metrics.MetricManager import MetricManager


class CacheMetadataManager:
    def __init__(self,dico_metadata,dico_data,dico_data_cache, formatter):
        self.dico_metadata = dico_metadata
        self.dico_data_cache = dico_data_cache
        self.dico_data = dico_data
        self.metric_manager = MetadataManager(dico_metadata,dico_data, formatter)
    def extract_from_list_of_possibilities(self,functions_list,dico_data):
        value = self.metric_manager.extract_from_list_of_possibilities(functions_list,dico_data)
        if value is None and self.dico_metadata["dataset"] == "ClassificationCache":
            value = self.metric_manager.extract_from_list_of_possibilities(functions_list,self.dico_data_cache)
        return value
    def extract_from_uniq_function(self,function,dico_data):
        value = self.metric_manager.extract_from_uniq_function(function,dico_data)
        if value is None and self.dico_metadata["dataset"] == "ClassificationCache":
            value = self.metric_manager.extract_from_uniq_function(function,self.dico_data_cache)
        return value
    def get(self,metadata_name):
        data = None
        if isinstance(self.dico_metadata[metadata_name]["access"],list):
            data = self.extract_from_list_of_possibilities(self.dico_metadata[metadata_name],self.dico_data)
        elif isinstance(self.dico_metadata[metadata_name]["access"],str):
            data = self.extract_from_uniq_function(self.dico_metadata[metadata_name],self.dico_data)
        else:
            print(self.dico_metadata[metadata_name])
            raise NotImplementedError
        if data is None:
            data = self.metric_manager.get_default(self.dico_metadata,metadata_name)
        return data