class TrValidSplit:
    def __init__(self,dataset,name):
        self.dataset = dataset
        self.name = name
    def __iter__(self):
        return self.dataset.__iter__(dataset=self.name)
    def len(self):
        return self.dataset.__len__(dataset=self.name)

def trvalidsplit(dataset):
    return [TrValidSplit(dataset,name="tr"),TrValidSplit(dataset,name="valid")]