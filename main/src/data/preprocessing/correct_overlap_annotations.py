import json
import re
from typing import Tuple, List

import dbf
import geopandas as gpd
import numpy as np
from PIL import Image, ImageDraw
from h5py import File
from rasterio.transform import Affine, rowcol

from main.FolderInfos import FolderInfos


def get_annotations():
    """ Extract annotation from the dbf table

    Returns: tuple of dicts:

    - annotations: provided as {idofannotation0: {"id": uniq_id.strip(), "label": type.strip(), "points": []}}
    - name_to_annotations: provided as {imageuniqid:[idofannotation0,...]}

    """
    FolderInfos.init(test_without_data=True)
    dbffile_path = FolderInfos.input_data_folder + "originals" + FolderInfos.separator + "Hydrocarbures_liquides_Seeps_et_spills_WGS84.dbf"
    annotations = {}
    name_to_annotations = {}
    with dbf.Table(dbffile_path) as table:  # Table containing the class and the index of the polygon
        for l in table:
            [id, id_img, id_nap, id_pol_nap, type, indice, sat, date, nom, mois, surface] = l
            nom = nom.split("\n")[0].strip()
            [_, t_init, t_end, uniq_id, preprocessing] = re.sub(
                "^(([0-9A-Za-z]+_){4})(\\d{8}T\\d{6})_(\\d{8}T\\d{6})_(([0-9A-Za-z]+_[0-9A-Za-z]+_[0-9A-Za-z]+))_([^\\.]+)(\\.data)?$",
                "\\1,\\3,\\4,\\5,\\7",
                nom
            ).split(",")
            annotations[id] = {"id": uniq_id.strip(), "label": type.strip(), "points": []}
            if uniq_id not in name_to_annotations:
                name_to_annotations[uniq_id] = []
            name_to_annotations[uniq_id].append(id)
    return annotations, name_to_annotations


def get_annotations_points(annotations, name_to_annotations):
    """

    Args:
        annotations: provided as {idofannotation0: {"id": uniq_id.strip(), "label": type.strip(), "points": []}}
        name_to_annotations: provided as {imageuniqid:[idofannotation0,...]}

    Returns: tuple of dict

    - annotations: provided as {idofannotation0: {"id": uniq_id.strip(), "label": type.strip(), "points": [[point1x,point1y],...]}}
    - name_to_annotations: provided as {imageuniqid:[idofannotation0,...]} (the same as the original)

    """
    ## Open the shapefile
    shapefile_path = FolderInfos.input_data_folder + "originals" + FolderInfos.separator + "Hydrocarbures_liquides_Seeps_et_spills_WGS84.shp"
    shapefile = gpd.read_file(shapefile_path)
    with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json", 'r') as fp:
        dico_informations = json.load(fp)
    for i_shape, shape in enumerate(shapefile.geometry):
        print(i_shape)
        id_shape = shapefile.id[i_shape]
        name = annotations[id_shape]["id"]
        transform_array = np.array(dico_informations[name]["transform"])
        transform = Affine.from_gdal(a=transform_array[0, 0], b=transform_array[0, 1], c=transform_array[0, 2],
                                     d=transform_array[1, 0], e=transform_array[1, 1], f=transform_array[1, 2])
        get_px_coord = lambda x, y: rowcol(transform, x, y)
        liste_points_shape: List[Tuple[int, int]] = []  # will contain the list of point of this shape
        elem = shape.boundary  # extract the boundary of the object shape (with other properties)
        if elem.geom_type != "LineString":  # the polygon is defined by a group of lines defines the polygon : https://help.arcgis.com/en/geodatabase/10.0/sdk/arcsde/concepts/geometry/shapes/types.htm
            # Utiliser le numéro de vertice pr éviter les croisements
            for line in elem:  # Loop through lines of the "Multi" - LineString
                coords = np.dstack(line.coords.xy).tolist()[0]  # get the list of points
                for point in coords:  # Extract the point of the polygon
                    px, py = get_px_coord(point[0], point
                    [1])  # Convert point coordinates from degrees to corresponding px coordinates
                    liste_points_shape.append(tuple([int(py), int(px)]))

        else:  # the polygon is defined one line which creates a closed shape
            coords = np.dstack(elem.coords.xy).tolist()[0]
            for point in coords:  # Extract the point of the polygon
                # Convert point coordinates from degrees to corresponding px coordinates
                px, py = get_px_coord(point[0], point[1])
                liste_points_shape.append(tuple([int(py), int(px)]))
        annotations[id_shape]["points"] = liste_points_shape
    return annotations, name_to_annotations


if __name__ == "__main__":

    annotations, name_to_annotations = get_annotations_points(*get_annotations())

    with File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r") as images_hdf5:
        with File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5",
                  "w") as annotations_labels_hdf5:

            for name in images_hdf5.keys():
                image_array = np.array(images_hdf5[name])
                # Shapefile Annotations map computation
                ## Create empty array with the same shape as the original image
                segmentation_map = np.zeros(shape=image_array.shape, dtype=np.uint8)
                segmentation_map = Image.fromarray(segmentation_map)
                draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
                try:
                    for index_shape in name_to_annotations[name]:
                        label = annotations[index_shape]["label"]
                        liste_points_shape = annotations[index_shape]["points"]
                        if label == "seep":  # Change color and so the value put in the array to create the label
                            color = "#010101"
                        elif label == "spill":
                            color = "#020202"
                        else:
                            color = "#000000"
                        draw.polygon(liste_points_shape, fill=color)
                except:
                    print(f"{name} has no annotation")
                    # Extract
                segmentation_map = np.array(segmentation_map, dtype=np.uint8)
                annotations_labels_hdf5.create_dataset(name, shape=segmentation_map.shape, dtype='i',
                                                       data=segmentation_map,
                                                       compression='gzip', compression_opts=9)
