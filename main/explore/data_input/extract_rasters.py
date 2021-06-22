import rasterio

from main.FolderInfos import FolderInfos
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def get_array_raster_file(path):
    with rasterio.open(path) as file_object:
        dataset = file_object.read(1)
    return dataset, file_object
if __name__ == "__main__":
    FolderInfos.init(test_without_data=True)
    files = [FolderInfos.data_test+f for f in os.listdir(FolderInfos.data_test)]
    dico_by_extensions = {}
    for sf,f in zip(os.listdir(FolderInfos.data_test),files):
        name = re.sub("^([A-Za-z0-9_]+[A-Za-z_]{,2})(20[0-9_A-Z]+)(\\.[a-z]+)$","\\2",sf)
        ext = sf.split(".")[-1]
        if ext not in dico_by_extensions:
            dico_by_extensions[ext] = {}
        dico_by_extensions[ext][name] = f
    for [[name,path],uniq_id] in zip(dico_by_extensions["img"].items(),["027481_0319CB_0EB7","016505_01F10F_CE84","016753_01F88A_4864"]):
        dataset, file_object = get_array_raster_file(path)
        # From https://gis.stackexchange.com/questions/311063/extract-raster-information-using-python
        radius_earth_meters = 6371e3
        xres = file_object.transform.a * np.pi/180.* radius_earth_meters
        yres = -file_object.transform.e * np.pi/180.* radius_earth_meters
        coordupperleft = file_object.transform * (0, 0)
        coordlowerright = file_object.transform * (file_object.width, file_object.height)
        bounds = file_object.bounds

        print(dataset.shape,xres,yres,bounds,file_object.crs)
        print(file_object.transform)
        plt.figure()
        plt.title(uniq_id)
        plt.imshow(dataset,cmap="gray")
        plt.savefig(FolderInfos.root_folder + "test_out"+FolderInfos.separator + "raster_open"+FolderInfos.separator +name+"_"+uniq_id+".png")
    plt.show()
    # cf https://rasterio.readthedocs.io/en/latest/quickstart.html#dataset-attributes
