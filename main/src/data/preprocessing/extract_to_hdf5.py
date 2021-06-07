
# Extract file names under the following structure

# {"shp":{"id":fullpath,...}, "img":, {"id":fullpath,...}}

import os, re
from main.FolderInfos import FolderInfos
FolderInfos.init(test_without_data=True)
shapefile_path = FolderInfos.input_data_folder +"originals" +FolderInfos.separator +"Hydrocarbures_liquides_Seeps_et_spills_WGS84.shp"
dbffile_path = FolderInfos.input_data_folder +"originals" +FolderInfos.separator +"Hydrocarbures_liquides_Seeps_et_spills_WGS84.dbf"
images_path_folder = FolderInfos.input_data_folder +"originals" +FolderInfos.separator +FolderInfos.separator.join \
    ("Sentinel1\\TraitementSnap".split("\\")) + FolderInfos.separator
shapefile_earth = FolderInfos.input_data_folder+"World map_Bufd.shp"

dico_by_extensions = {"img" :{}}
dico_infos_img = {}

for folder in os.listdir(images_path_folder):
    img_file_path = images_path_folder +folder +FolderInfos.separator +"Sigma0_VV_db.img"
    # Extract the name of the file
    [_ ,t_init ,t_end ,uniq_id ,preprocessing] = re.sub(
        "^(([0-9A-Za-z]+_){4})(\\d{8}T\\d{6})_(\\d{8}T\\d{6})_(([0-9A-Za-z]+_[0-9A-Za-z]+_[0-9A-Za-z]+))_([^\\.]+)\\.data$",
        "\\1,\\3,\\4,\\5,\\7",
        folder
        ).split(",")
    dico_by_extensions["img"][uniq_id] = img_file_path
    dico_infos_img[uniq_id] = {"t_init" :t_init ,"t_end" :t_end ,"uniq_id" :uniq_id ,"preprocessing" :preprocessing}

# Write the following files:

"""
- images.hdf5: future main inputs of the network, numpy arrays of shape (height,width,1) dtype = np.float32
- annotations_labels.hdf5: reference for the output of the segmentation network: numpy array of shape (height,width,3) dtype = np.uint8 !.

The value 3 correspond respectively (in this order) to oil seep, oil spill and others.
- class_mappings.json: will store mapping between class names (oil seep, oil spill and others) and index in the array of the annotations_labels.hdf5 file (this file is created manually in the data_in folder)
- images_informations.json: will store
    - coord_upperleft and coord_lowerright: floats tuples. Coordinates of the upper left and lower right corners in degrees
    - resolution: floats tuple: Its resolution in px/meters
    - tinit,tend: initial and ending time when satelite start to capture
    - preprocessing: code of preprocessing steps

All data will be accessible by calling object[image_id]
"""
from typing import Tuple, List

from h5py import File
import json

FolderInfos.init(test_without_data=True)
def get_mode(path):
    mode = "r+"
    if os.path.exists(path) is False:
        mode = "w"
    return mode
already_done = None
## Create objects hdf5 and container for th images informations
if os.path.exists(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5") is True:
    with File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5" ,"r") as images_hdf5:
        already_done = list(images_hdf5.keys())
path_img = f"{FolderInfos.input_data_folder}images_preprocessed.hdf5"
mode = get_mode(path_img)
images_hdf5 = File(path_img ,mode)
annotations_labels_hdf5 = File(f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5" ,mode)
images_informations = {}

import geopandas as gpd
from shapely import speedups
import rasterio
import numpy as np
from PIL import Image ,ImageDraw # PIL = pillow
import dbf

speedups.disable() # To avoid errors
#
# # Extract earth land points
# land_point_list = []
# shapefile_land = gpd.read_file(shapefile_earth)
# for shape in shapefile_land.geometry:
#     liste_points_shape: List[Tuple[int ,int]] = [] # will contain the list of point of this shape
#     elem = shape.boundary # extract the boundary of the object shape (with other properties)
#     if elem.geom_type != "LineString"  :# the polygon is defined by a group of lines defines the polygon : https://help.arcgis.com/en/geodatabase/10.0/sdk/arcsde/concepts/geometry/shapes/types.htm
#         # Utiliser le numéro de vertice pr éviter les croisements
#         for line in elem: # Loop through lines of the "Multi" - LineString
#             coords = np.dstack(line.coords.xy).tolist()[0] # get the list of points
#             for point in coords: # Extract the point of the polygon
#                 land_point_list.append(tuple([point[0], point[1]]))
#
#     else: # the polygon is defined one line which creates a closed shape
#         coords = np.dstack(elem.coords.xy).tolist()[0]
#         for point in coords: # Extract the point of the polygon
#             # Convert point coordinates from degrees to corresponding px coordinates
#             px, py = raster.index(point[0], point[1])
#             liste_points_shape.append(tuple([int(py), int(px)]))



## Open the shapefile
shapefile = gpd.read_file(shapefile_path)
## Open corresponding database storing
table = dbf.Table(dbffile_path)  # Table containing the class and the index of the polygon
table.open()
## Loop through images, open them and add their informations to the correspinding objects
nb_elems = len(dico_by_extensions["img"])
for i ,[name ,pathImg] in enumerate(dico_by_extensions["img"].items()):
    print(f"Progress {( i +1 ) /nb_elems*100:.2f}%")
    if already_done is not None and name in already_done:
        print(f"{name} has already been done")
        continue # We skip this image
    # We loop through raster images and shapefiles
    ## Open the raster
    with rasterio.open(pathImg) as raster: ## (NB: with keyword allows to manage files (properly open and close them)
        image_array: np.ndarray = raster.read(1) # Get the image array
        # Then, we create a "dataset" to be able to access the data by calling hdf5_object[name]. It is not a real dataset as we commonly think as only one image is in it.
        images_hdf5.create_dataset(name ,shape=image_array.shape ,dtype='f',
                                   data=image_array, compression='gzip', compression_opts=9)
        # Properties computation
        ## Resolution computation
        # From https://gis.stackexchange.com/questions/311063/extract-raster-information-using-python
        radius_earth_meters = 6371e3
        # raster.transform is the transformation matrix between the spatial coordinates and the pixels # https://en.wikipedia.org/wiki/Transformation_matrix#:~:text=eigenbases-,examples%20in%202%20dimensions,-edit
        xres: float = abs \
            (raster.transform.a * np.pi / 180. * radius_earth_meters) # We consider that the earth is a sphere
        yres: float = abs \
            (raster.transform.e * np.pi / 180. * radius_earth_meters) # abs in case of a reflexion transformation
        ## Coordinates
        coord_upperleft: Tuple[float ,float] = raster.transform * (0, 0)
        coord_lowerright: Tuple[float ,float] = raster.transform * (raster.width, raster.height)
        images_informations[name] = {"resolution" :(xres ,yres) ,"coord_upperleft" :coord_upperleft
                                     ,"coord_lowerright" :coord_lowerright,
                                     **dico_infos_img[name]} # **dictionary = dictionnary unpacking: we add the content of the dict to the other dict

    # Shapefile segmentation map computation
    ## Create empty array with the same shape as the original image
    segmentation_map = np.zeros(shape=image_array.shape ,dtype=np.uint8)
    segmentation_map = Image.fromarray(segmentation_map)
    draw = ImageDraw.ImageDraw(segmentation_map)  # draw the base image
    for i_shape ,shape in enumerate(shapefile.geometry):
        liste_points_shape: List[Tuple[int ,int]] = [] # will contain the list of point of this shape
        elem = shape.boundary # extract the boundary of the object shape (with other properties)
        if elem.geom_type != "LineString"  :# the polygon is defined by a group of lines defines the polygon : https://help.arcgis.com/en/geodatabase/10.0/sdk/arcsde/concepts/geometry/shapes/types.htm
            # Utiliser le numéro de vertice pr éviter les croisements
            for line in elem: # Loop through lines of the "Multi" - LineString
                coords = np.dstack(line.coords.xy).tolist()[0] # get the list of points
                for point in coords: # Extract the point of the polygon
                    px, py = raster.index(point[0], point
                        [1]) # Convert point coordinates from degrees to corresponding px coordinates
                    liste_points_shape.append(tuple([int(py), int(px)]))

        else: # the polygon is defined one line which creates a closed shape
            coords = np.dstack(elem.coords.xy).tolist()[0]
            for point in coords: # Extract the point of the polygon
                # Convert point coordinates from degrees to corresponding px coordinates
                px, py = raster.index(point[0], point[1])
                liste_points_shape.append(tuple([int(py), int(px)]))
        id_shape = shapefile.id[i_shape]
        metadata = list(filter(lambda x: x[0] == id_shape, table))[0]
        label = metadata[4].strip() # strip cut all space, back to line
        if label == "seep":# Change color and so the value put in the array to create the label
            color  = "#010101"
        elif label == "spill":
            color  = "#020202"
        else:
            color = "#000000"
        draw.polygon(liste_points_shape, fill=color)
        # Extract
    segmentation_map = np.array(segmentation_map ,dtype=np.uint8)
    annotations_labels_hdf5.create_dataset(name ,shape=segmentation_map.shape ,dtype='i' ,data=segmentation_map,
                                           compression='gzip', compression_opts=9)

# Write the image informations to the corresponding file
with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json",mode) as fp: # NB: fp = filepointer
    previous_images_informations = json.load(fp) if mode != "w" else {}
    previous_images_informations.update(images_informations)
    fp.seek(0)
    json.dump(previous_images_informations ,fp,indent=4)
table.close()
images_hdf5.close()
annotations_labels_hdf5.close()
