from main.src.data.segmentation.DataSentinel1Segmentation import DataGet


class DatasetFactory:
    def __init__(self,dataset_name="sentinel1", usage_type="classification"):
        if usage_type == "classification":
            if dataset_name == "sentinel1":
                self.dataset = # TODO
        elif usage_type == "segmentation":
            if dataset_name == "sentinel1":
                self.dataset = DataGet()
        else:
            raise Exception(f"{usage_type} not supported")
    def __call__(self):
        return self.dataset