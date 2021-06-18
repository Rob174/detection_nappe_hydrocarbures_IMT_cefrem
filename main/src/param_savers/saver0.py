from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.param_savers.BaseClass import BaseClass
import json


class Saver0:
    def __init__(self,outpath):
        self.outpath = outpath
        self.data = {}
    def recursive_dict(self,object):
        if isinstance(object,BaseClass) is False:
            return object
        else:
            dico_params = {}
            for attr,val in object.__dict__.items():
                if attr[:4] == "attr":
                    dico_params[attr] = self.recursive_dict(val)
        return dico_params
    def __call__(self, object):
        if isinstance(object,BaseClass) is False:
            return object
        name = object.attr_global_name
        dico_params = self.recursive_dict(object)
        self.data[name] = dico_params
        return self
    def __setitem__(self, key, value):
        self.data[key] = value
    def save(self):
        with open(self.outpath,"w") as fp:
            json.dump(self.data,fp,indent=4)

if __name__ == "__main__":
    class Test(BaseClass):
        def __init__(self, a, b, c):
            self.attr_a = a
            self.attr_b = b
            self.c = c
    t = Test(1,Test(4,5,6),3)
    s = Saver0("")(t)
    print(s)



    FolderInfos.init(test_without_data=False)
    dataset_factory = DatasetFactory(dataset_name="sentinel1", usage_type="classification", patch_creator="fixed_px",
                                     patch_padding="no", grid_size=1000, input_size=256)
    print(Saver0("")(dataset_factory))