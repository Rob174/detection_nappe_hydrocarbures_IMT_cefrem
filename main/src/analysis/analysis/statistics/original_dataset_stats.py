import json

from h5py import File

from main.FolderInfos import FolderInfos

if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    with File(FolderInfos.input_data_folder+"images_preprocessed.hdf5","r") as cache:
        images = list(cache.keys())
    extract = lambda x:"_".join(x.split("_")[6:9])
    with open(FolderInfos.input_data_folder+"images_stat_seep.txt","r") as fp:
        seeps = {extract(l.strip().split("\t")[0]):l.strip().split("\t")[1] for l in fp.readlines()[1:]}
    with open(FolderInfos.input_data_folder+"images_stat_spill.txt","r") as fp:
        spills = {extract(l.strip().split("\t")[0]):l.strip().split("\t")[1] for l in fp.readlines()[1:]}
    stats = {}
    for name in images:
        combination = ""
        if name in seeps:
            combination += str(seeps[name])
        else:
            combination += "0"
        combination += ","
        if name in spills:
            combination += str(spills[name])
        else:
            combination += "0"
        if combination not in stats:
            stats[combination] = 0
        stats[combination] += 1

    with open(FolderInfos.input_data_folder+"qgis_type_stats_images_preprocessed.json","w") as fp:
        json.dump(stats,fp)
