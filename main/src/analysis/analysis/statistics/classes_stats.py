import json

from main.FolderInfos import FolderInfos
from main.src.data.Datasets.Fabrics.FabricFilteredCache import FabricFilteredCache
from main.src.data.LabelModifier.LabelModifier0 import LabelModifier0
from main.src.data.LabelModifier.LabelModifier1 import LabelModifier1
from main.src.data.LabelModifier.LabelModifier2 import LabelModifier2

if __name__ == '__main__':
    FolderInfos.init(test_without_data=True)
    images,annotations,infos = FabricFilteredCache()()
    name_file = "preprocessed_cache"
    label_maker = LabelModifier0(class_mapping=annotations.attr_mapping)
    dico_stats = {}
    name_class = None
    for name in annotations.keys():
        annotation = annotations.get(name)
        label = label_maker.make_classification_label(annotation=annotation)
        name_class = []
        if label[annotations.attr_mapping["other"]] == 1:
            name_class.append("other")
        if label[annotations.attr_mapping["seep"]] == 1:
            name_class.append("seep")
        if label[annotations.attr_mapping["spill"]] == 1:
            name_class.append("spill")
        name_class = "_".join(sorted(name_class))
        if name_class not in dico_stats.keys():
            dico_stats[name_class] = 0
        dico_stats[name_class] += 1
    with open(FolderInfos.input_data_folder+name_file+"classes_stats.json","w") as fp:
        json.dump(dico_stats,fp)


