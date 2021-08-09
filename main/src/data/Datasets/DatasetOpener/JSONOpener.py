import json

from main.src.data.Datasets.DatasetOpener.AbstractOpener import AbstractOpener


class JSONOpener(AbstractOpener):
    def __init__(self, path: str):
        super(JSONOpener, self).__init__(path)
        with open(self.attr_path, "r") as fp:
            self.dataset = json.load(fp)