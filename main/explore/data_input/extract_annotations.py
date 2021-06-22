from typing import Tuple, List

import os
from main.FolderInfos import FolderInfos
import re
from PIL import ImageDraw, Image
from main.explore.data_input.extract_rasters import get_array_raster_file
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely import speedups
import dbf

if __name__ == "__main__":
    speedups.disable()

    FolderInfos.init(test_without_data=True)
    # get files names
    files = [FolderInfos.data_test+f for f in os.listdir(FolderInfos.data_test)]
    dico_by_extensions = {}
    for sf,f in zip(os.listdir(FolderInfos.data_test),files):
        # Extract the name of the file
        name = re.sub("^([A-Za-z0-9_]+[A-Za-z_]{,2})(20[0-9_A-Z]+)(\\.[a-z]+)$","\\2",sf)
        ext = sf.split(".")[-1]
        if ext not in dico_by_extensions:
            dico_by_extensions[ext] = {}
        # Split files by extension (.shp, .img....) and by names
        dico_by_extensions[ext][name] = f



    for [[name, pathShp],[_,pathImg],[_,pathDbf]] in zip(dico_by_extensions["shp"].items(),dico_by_extensions["img"].items(),dico_by_extensions["dbf"].items()):
        # We loop through raster images and shapefiles
        shp = gpd.read_file(pathShp) # Open the shapefile
        array,raster_object = get_array_raster_file(pathImg) # Get raster file
        transform_array = raster_object.transform # Get the transformation matrix
        points_list: List[List[Tuple[int,int]]] = [] # Will contain the list of points for each polygon of the shapefile
        # Convert image to rgb gray scale for pillow (used to make the overlay of oil discharges)
        array = np.stack((array,)*3,axis=-1)
        array = (array - np.min(array))/(np.max(array)-np.min(array)) # Convert to range between 0-1 to be able to plot the output image
        print(array.shape,array.dtype,np.min(array),np.max(array))
        img = Image.fromarray((array*255).astype(np.uint8)) # Convert to 0-255 uint images
        img2 = img.copy() # Copy as Pillow modifies the input, to be able to make the overlay
        draw = ImageDraw.ImageDraw(img2) # draw the base image

        ## Open corresponding database storing
        table = dbf.Table(pathDbf) # Table containing the class and the index of the polygon
        table.open()
        for i_shape,shape in enumerate(shp.geometry):
            liste_points_shape: List[Tuple[int,int]] = [] # will contain the list of point of this shape
            elem = shape.boundary # extract the boundary of the object shape (with other properties)
            if elem.geom_type != "LineString":# a group of lines defines the polygon : # https://help.arcgis.com/en/geodatabase/10.0/sdk/arcsde/concepts/geometry/shapes/types.htm
                # Utiliser le numéro de vertice pr éviter les croisements
                for line in elem: # Loop through lines of the "Multi" - LineString
                    coords = np.dstack(line.coords.xy).tolist()[0] # get the list of points
                    for point in coords: # Extract the point of the polygon
                        px, py = raster_object.index(point[0], point[1]) # Convert point coordinates from degrees to corresponding px coordinates
                        liste_points_shape.append(tuple([int(px), int(py)]))

            else: # closed shape
                coords = np.dstack(elem.coords.xy).tolist()[0]
                for point in coords: # Extract the point of the polygon
                    px, py = raster_object.index(point[0], point[1]) # Convert point coordinates from degrees to corresponding px coordinates
                    liste_points_shape.append(tuple([int(py), int(px)]))

            print(liste_points_shape)
            id_shape = shp.id[i_shape]
            metadata = list(filter(lambda x:x[0] == id_shape,table))[0]
            label = metadata[4]
            points_list.append(liste_points_shape) # add the list of points of the current shape to the global list containing points of all shapes
            draw.polygon(liste_points_shape, fill="red") # draw the polygon on the image
        table.close()
        img3 = Image.blend(img, img2, 0.5) # Show it overlayed on the image with a "blending factor" of 50%
        plt.figure(figsize=(10,10))
        plt.title(name)
        plt.imshow(img3)
        plt.savefig(FolderInfos.root_folder + "test_out"+FolderInfos.separator + "raster_annotated"+FolderInfos.separator +name+"_annotated.png")
    # plt.show()