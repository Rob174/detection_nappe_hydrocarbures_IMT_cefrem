class DefaultValueNotFound(Exception):
    pass


class MetadataManager:
    def __init__(self, dico_metadata, dico_data, formatter):
        self.dico_metadata = dico_metadata
        self.dico_data = dico_data
        self.formatter = formatter

    def extract_from_list_of_possibilities(self, functions_list, dico_data):
        for function in functions_list:
            try:
                value = self.formatter.format(eval(function)(dico_data))
                return value
            except:
                pass
        return None

    def extract_from_uniq_function(self, function, dico):
        try:
            value = self.formatter.format(eval(function)(dico))
            return value
        except:
            return None

    def get_default(self, dico_metadata, metadata_name):
        try:
            value = eval(dico_metadata[metadata_name]["default"])
        except KeyError:
            raise DefaultValueNotFound
        return value

    def get(self, metadata_name):
        data = None
        if isinstance(self.dico_metadata[metadata_name]["access"], list):
            data = self.extract_from_list_of_possibilities(self.dico_metadata[metadata_name]["access"], self.dico_data)
        elif isinstance(self.dico_metadata[metadata_name]["access"], str):
            data = self.extract_from_uniq_function(self.dico_metadata[metadata_name]["access"], self.dico_data)
        else:
            raise NotImplementedError
        if data is None:
            data = self.get_default(self.dico_metadata, metadata_name)
        return data
