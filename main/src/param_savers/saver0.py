import json

from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.param_savers.BaseClass import BaseClass


class Saver0:
    """Object used to automatically save attributes of classes beginning by attr_... recursively:
    if an object has an attr_ attribute and this attribute is a class inherited from the BaseClass, the saver will also scan attributes in this class

    Args:
        outpath: str, outpath of the json file
    """

    def __init__(self, outpath):
        """


        """
        self.outpath = outpath
        self.data = {}

    def recursive_dict(self, object):
        """Function scanning an attribute. if this is a class inheriting from BaseClass,
        it put all attr_... attributes in a dict by recursively calling recursive_dict

        Args:
            object: a simple value (int,str,list,dict) or a class inheriting from BaseClass

        Returns:
            the value of the attribute (if it is a simple value) or a dict of values

        """
        if (isinstance(object, list) or isinstance(object, tuple)) and (
                len(object) > 0 and isinstance(object[0], BaseClass)):
            return [self.recursive_dict(o) for o in object]
        if isinstance(object, BaseClass) is False:
            return object
        else:
            dico_params = {}
            for attr, val in object.__dict__.items():
                if attr[:4] == "attr":
                    dico_params[attr] = self.recursive_dict(val)
        return dico_params

    def __call__(self, object):
        return self.call(object)

    def call(self, object):
        """Add all attr_... attributes of this object under the object.attr_global_name field

        Args:
            object: object of class inheriting from BaseClass

        Returns:
            the current object (to eventually chain the call)

        """
        if isinstance(object, BaseClass) is False:
            return object
        name = object.attr_global_name
        dico_params = self.recursive_dict(object)
        self.data[name] = dico_params
        return self

    def setitem(self, key, value):
        """Method from the magic method

        Args:
            key: name of the key where to put data in the root dict
            value: value to add

        Returns:

        """
        self.data[key] = value

    def __setitem__(self, key, value):
        self.setitem(key, value)

    def save(self):
        """Save data to the json file specified in the constructor"""
        with open(self.outpath, "w") as fp:
            json.dump(self.data, fp, indent=4)


if __name__ == "__main__":
    class Test(BaseClass):
        def __init__(self, a, b, c):
            self.attr_a = a
            self.attr_b = b
            self.c = c


    t = Test(1, Test(4, 5, 6), 3)
    s = Saver0("")(t)
    print(s)

    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1", usage_type="classification", patch_creator="fixed_px",
                                     patch_padding="no", grid_size=1000, input_size=256)
    print(Saver0("")(dataset_factory))
