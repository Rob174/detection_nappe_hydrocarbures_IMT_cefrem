from enum import Enum

from main.src.param_savers.BaseClass import BaseClass

class Way(Enum):
    ORIGINAL_WAY = "original_way"
    OTHER_WAY = "other_way"
class TwoWayDict(BaseClass):
    def __init__(self,attr_dico_one_way,additionnal_infos_attached=None):
        self.attr_global_name = "two_way_dict"
        if additionnal_infos_attached is None:
            self.attr_dico_one_way = attr_dico_one_way
            self.dico_other_way = {v:k for k,v in attr_dico_one_way.items()}
        else:
            self.attr_dico_one_way = {k:(v,additionnal_infos_attached[k]) for k,v in attr_dico_one_way.items()}
            self.dico_other_way = {v:(k,additionnal_infos_attached[k]) for k,v in attr_dico_one_way.items()}
    def __getitem__(self, item):
        try:
            return self.attr_dico_one_way[item]
        except:
            return self.dico_other_way[item]
    def __setitem__(self, key, value):
        value_key = value
        if isinstance(value,tuple):
            value_key,additionnal_info = value
        real_key,dico_chosen = key
        if dico_chosen == Way.ORIGINAL_WAY:
            self.attr_dico_one_way[real_key] = value
            self.dico_other_way[value_key] = real_key
        else:
            self.attr_dico_one_way[value_key] = real_key
            self.dico_other_way[real_key] = value
    def keys(self,dico_chosen=Way.ORIGINAL_WAY):
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.keys())
        else:
            return list(self.dico_other_way.keys())
    def values(self,dico_chosen=Way.ORIGINAL_WAY):
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.values())
        else:
            return list(self.dico_other_way.values())
    def items(self,dico_chosen=Way.ORIGINAL_WAY):
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.items())
        else:
            return list(self.dico_other_way.items())
    def __len__(self):
        return len(list(self.attr_dico_one_way.keys()))