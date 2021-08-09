import pickle

from main.src.data.Datasets.DatasetOpener.AbstractOpener import AbstractOpener


class PickleOpener(AbstractOpener):

    def __init__(self, path: str):
        super(PickleOpener, self).__init__(path)
        with open(path, "rb") as fp:
            self.dataset = pickle.load(fp)