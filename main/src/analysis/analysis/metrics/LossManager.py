from main.src.analysis.analysis.exceptions import ElementNotFound


class LossManager:
    def __init__(self):
        pass

    def reformat(self, dico_values):
        tr_values = self.reformat_dataset(dico_values, "tr")
        valid_values = self.reformat_dataset(dico_values, "valid")
        return {"tr_values": tr_values, "valid_values": valid_values}

    def reformat_dataset(self, dico_values, dataset="tr"):
        values = None
        if f"attr_{dataset}_loss" in dico_values:
            values = dico_values[f"attr_{dataset}_loss"]
        elif dataset in dico_values:
            values = dico_values[dataset]
        else:
            raise ElementNotFound
        return values
