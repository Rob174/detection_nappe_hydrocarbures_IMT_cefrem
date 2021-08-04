import json

import geopandas as gpd
import numpy as np
from rasterio.transform import Affine, rowcol

from main.FolderInfos import FolderInfos
import pickle
if __name__ == '__main__':
    dbffile_path = r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\originals\filtered_world.dbf"
    shapefile_path = r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\originals\filtered_world.shp"

    FolderInfos.init(test_without_data=True)
    shapes = []
    shapefile = gpd.read_file(shapefile_path)
    for i_shape, shape in enumerate(shapefile.geometry):
        print(i_shape,end="\r")
        elem = shape.boundary  # extract the boundary of the object shape (with other properties)
        if elem.geom_type != "LineString":  # the polygon is defined by a group of lines defines the polygon : https://help.arcgis.com/en/geodatabase/10.0/sdk/arcsde/concepts/geometry/shapes/types.htm
            # Utiliser le numéro de vertice pr éviter les croisements
            points = []
            for line in elem:  # Loop through lines of the "Multi" - LineString
                coords = np.dstack(line.coords.xy).tolist()[0]  # get the list of points
                for point in coords:  # Extract the point of the polygon
                    points.append(tuple([*point]))
            shapes.append(points)

        else:  # the polygon is defined one line which creates a closed shape
            coords = np.dstack(elem.coords.xy).tolist()[0]
            points = []
            for point in coords:  # Extract the point of the polygon
                points.append(tuple([*point]))
            shapes.append(points)
    with open(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\preprocessed_cache\images_informations_preprocessed.json","r") as fp:
        infos = json.load(fp)
    dico_data = {}
    for i,[id,dico] in enumerate(infos.items()):
        print(i,end="\r")
        transform_array = np.array(dico["transform"])
        transform = Affine.from_gdal(a=transform_array[0, 0], b=transform_array[0, 1], c=transform_array[0, 2],
                                     d=transform_array[1, 0], e=transform_array[1, 1], f=transform_array[1, 2])
        get_px_coord = lambda x, y: rowcol(transform, x, y)
        dico_data[id] = []
        for shape in shapes:
            l=[]
            for point in shape:
                l.append(get_px_coord(*point))
            if np.max(l) > 0:
                dico_data[id].append(l)
    with open(FolderInfos.input_data_folder+"preprocessed_cache"+FolderInfos.separator+"world_earth_data.pkl","wb") as fp:
        pickle.dump(dico_data,fp)
