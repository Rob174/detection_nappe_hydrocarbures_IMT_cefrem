import rasterio

import sys
import os
from main.FolderInfos import FolderInfos
import re
from PIL import ImageDraw, Image
from main.explore.data_input.extract_rasters import get_array_raster_file
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely import speedups

speedups.disable()

FolderInfos.init(test_without_data=True)
files = [FolderInfos.data_test+f for f in os.listdir(FolderInfos.data_test)]
dico_by_extensions = {}
for sf,f in zip(os.listdir(FolderInfos.data_test),files):
    name = re.sub("^([A-Za-z0-9_]+[A-Za-z_]{,2})(20[0-9_A-Z]+)(\\.[a-z]+)$","\\2",sf)
    ext = sf.split(".")[-1]
    if ext not in dico_by_extensions:
        dico_by_extensions[ext] = {}
    dico_by_extensions[ext][name] = f



for [[name, pathShp],[_,pathImg]] in zip(dico_by_extensions["shp"].items(),dico_by_extensions["img"].items()):
    shp = gpd.read_file(pathShp)
    array,raster_object = get_array_raster_file(pathImg)
    transform_array = raster_object.transform
    points_list = []
    array = np.stack((array,)*3,axis=-1)
    array = (array - np.min(array))/(np.max(array)-np.min(array))
    print(array.shape,array.dtype,np.min(array),np.max(array))
    img = Image.fromarray((array*255).astype(np.uint8))
    img2 = img.copy()
    draw = ImageDraw.ImageDraw(img2)

    g = [i for i in shp.geometry]
    nb = 0
    for shape in g:
        liste_points_px = []
        elem = shape.boundary
        if elem.geom_type != "LineString":
            for line in elem:
                coords = np.dstack(line.coords.xy).tolist()[0]
                for point in coords:
                    px, py = raster_object.index(point[0], point[1])
                    liste_points_px.append(tuple([int(px), int(py)]))

        else:
            coords = np.dstack(elem.coords.xy).tolist()[0]
            for point in coords:
                px, py = raster_object.index(point[0], point[1])
                liste_points_px.append(tuple([int(px),int(py)]))

        print(liste_points_px)
        points_list.append(liste_points_px)
        draw.polygon(liste_points_px, fill="wheat")
    img3 = Image.blend(img, img2, 0.5)
    plt.figure()
    plt.imshow(img3)
plt.show()