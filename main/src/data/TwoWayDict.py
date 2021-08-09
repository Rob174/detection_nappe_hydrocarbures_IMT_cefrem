"""An object allowing to map two values together and access to them directly as in a dict. It can contains additionnal informations linked to pairs"""
from enum import Enum

from main.src.param_savers.BaseClass import BaseClass


class Way(Enum):
    ORIGINAL_WAY = "original_way"
    OTHER_WAY = "other_way"


class TwoWayDict(BaseClass):
    """
    An object allowing to map two values together and access to them directly as in a dict. It can contains additionnal informations linked to pairs

    ⚠️ Not tested for partial informations: if we provide one additionnal information for one pair, we need to do it for all pairs

    Args:
        attr_dico_one_way: dict, dictionary containing the mapping with additionnal informations in the value field


    Example 1: creating a TwoWayDict and getting values

            >>> dico = TwoWayDict({"a":1,"b":2,"c":3})
            >>> dico["a"] # as for a normal dict we can get values as follows
            1
            >>> dico[1] # but we can also access the mapped property
            "a"

    Example 2: you can also use the traditionnal .keys() .values() and .items() and specify which keys to use:

            >>> dico.keys(Way.ORIGINAL_WAY) # as for a normal dict we can get values as follows
            ["a","b","c"]
            >>> dico.keys(Way.OTHER_WAY) # as for a normal dict we can get values as follows
            [1,2,3]
            >>> dico.items(Way.ORIGINAL_WAY) # as for a normal dict we can get values as follows
            [("a",1),("b",2),("c",3)]
            >>> dico.keys(Way.OTHER_WAY) # as for a normal dict we can get values as follows
            [(1,"a"),(2,"b"),(3,"c")]

    Example 3: this object also allows to store additionnal informations with original data

            >>> dico = TwoWayDict({"a":(1,"info1"),"b":(2,"info2"),"c":(3,"info3")})
            >>> dico[1]
            "a","info1"
            >>> dico["a"]
            1,"info1"

    Example 4: we can also insert values by providing to which group belongs the key:

            >>> dico[4,Way.OTHER_WAY] = "d","info4"

    """

    def __init__(self, attr_dico_one_way):
        self.attr_global_name = "two_way_dict"
        if attr_dico_one_way == {}:
            self.attr_dico_one_way = {}
            self.dico_other_way = {}
        else:
            first_key = attr_dico_one_way[list(attr_dico_one_way.keys())[0]]
            if isinstance(first_key, tuple):
                self.attr_dico_one_way = {}
                self.dico_other_way = {}
                for k, v in attr_dico_one_way.items():
                    self.attr_dico_one_way[k] = v
                    self.dico_other_way[v[0]] = k, *v[1:]
            else:
                self.attr_dico_one_way = attr_dico_one_way
                self.dico_other_way = {v: k for k, v in attr_dico_one_way.items()}

    def __getitem__(self, item):
        return self.getitem(item)

    def getitem(self, item):
        """Get an item in the TwoWayDict trying first in the first provided way and then in the reverse way

        Args:
            item: key for which to get the corresponding value

        Returns: the corresponding value to this key with if provided additionnal informations (cf class doc examples)

        """
        try:
            return self.attr_dico_one_way[item]
        except:
            return self.dico_other_way[item]

    def __setitem__(self, key, value):
        return self.setitem(key, value)

    def setitem(self, key, value):
        """Add a new value or change the value of key provided.

        Args:
            key: ⚠️ tuple, providing both the key for which you want to set the value and the information to which way the key belongs (as enum Way)
            value: the value corresponding to this key with additionnal informations. If so provide a tuple value

        Returns:

        """
        value_key = value
        real_key, dico_chosen = key
        value1 = real_key
        if isinstance(value, tuple):
            value_key = value[0]
            additionnal_info = value[1:]
            value1 = real_key, *additionnal_info
        if dico_chosen == Way.ORIGINAL_WAY:
            self.attr_dico_one_way[real_key] = value
            self.dico_other_way[value_key] = value1
        else:
            self.attr_dico_one_way[value_key] = value1
            self.dico_other_way[real_key] = value

    def keys(self, dico_chosen=Way.ORIGINAL_WAY):
        """The keys of the object in the way asked

        Args:
            dico_chosen: enum Way

        Returns: the list of keys of the dict in the original way or in the other way

        """
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.keys())
        else:
            return list(self.dico_other_way.keys())

    def values(self, dico_chosen=Way.ORIGINAL_WAY):
        """The values of the object in the way asked

        Args:
            dico_chosen:  enum Way

        Returns: the list of values of the dict in the original way or in the other way

        """
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.values())
        else:
            return list(self.dico_other_way.values())

    def items(self, dico_chosen=Way.ORIGINAL_WAY):
        """The items of the object in the way asked

        Args:
            dico_chosen: enum Way

        Returns: the list of tuples (key,val) of the dict in the original way or in the other way

        """
        if dico_chosen == Way.ORIGINAL_WAY:
            return list(self.attr_dico_one_way.items())
        else:
            return list(self.dico_other_way.items())

    def __len__(self):
        return len(list(self.attr_dico_one_way.keys()))
    def __str__(self):
        return str(self.attr_dico_one_way)