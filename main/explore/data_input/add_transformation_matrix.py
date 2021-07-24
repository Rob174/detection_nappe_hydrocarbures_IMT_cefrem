
# Extract file names under the following structure

# {"shp":{"id":fullpath,...}, "img":, {"id":fullpath,...}}
if __name__ == "__main__":
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
    - annotations_labels.hdf5: reference for the output of the Fabrics network: numpy array of shape (height,width,3) dtype = np.uint8 !.
    
    The value 3 correspond respectively (in this order) to oil seep, oil spill and others.
    - class_mappings.json: will store mapping between class names (oil seep, oil spill and others) and index in the array of the annotations_labels.hdf5 file (this file is created manually in the data_in folder)
    - images_informations.json: will store
        - coord_upperleft and coord_lowerright: floats tuples. Coordinates of the upper left and lower right corners in degrees
        - resolution: floats tuple: Its resolution in px/meters
        - tinit,tend: initial and ending time when satelite start to capture
        - preprocessing: code of preprocessing steps
    
    All data will be accessible by calling object[image_id]
    """

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
    path_annotations = f"{FolderInfos.input_data_folder}annotations_labels_preprocessed.hdf5"
    mode = get_mode(path_annotations)
    annotations_labels_hdf5 = File(path_annotations ,mode)
    images_informations = {}

    from shapely import speedups
    import rasterio
    import numpy as np

    speedups.disable() # To avoid errors

    ## Loop through images, open them and add their informations to the correspinding objects
    nb_elems = len(dico_by_extensions["img"])
    with open(f"{FolderInfos.input_data_folder}images_informations_preprocessed.json","r+") as fp: # NB: fp = filepointer
        previous_images_informations = json.load(fp)
        for i ,[name ,pathImg] in enumerate(dico_by_extensions["img"].items()):
            print(f"Progress {( i +1 ) /nb_elems*100:.2f}%")
            # We loop through raster images and shapefiles
            ## Open the raster
            with rasterio.open(pathImg) as raster: ## (NB: with keyword allows to manage files (properly open and close them)
                matrix = np.array(raster.transform).reshape(-1,3).tolist()

                previous_images_informations[name]["transform"] = matrix
        fp.seek(0)
        json.dump(previous_images_informations ,fp,indent=4)
