import pickle
from enum import Enum

from h5py import File

from main.FolderInfos import FolderInfos
from main.src.data.preprocessing.correct_overlap_annotations import get_annotations_points, get_annotations

class EnumShapeCategories(str,Enum):
    Label = "label"
    Points = "poitns"

if __name__ == "__main__":

    annotations, name_to_annotations = get_annotations_points(*get_annotations())

    with File(f"{FolderInfos.input_data_folder}images_preprocessed.hdf5", "r") as images_hdf5:

            reformatted_dico = {}
            for name in images_hdf5.keys():
                reformatted_dico[name] = []
                try:
                    for index_shape in name_to_annotations[name]:
                        label = annotations[index_shape][EnumShapeCategories.Label]
                        liste_points_shape = annotations[index_shape][EnumShapeCategories.Points]
                        reformatted_dico[name].append({EnumShapeCategories.Label:label,EnumShapeCategories.Points:liste_points_shape})
                except KeyError:
                    pass
            with open(FolderInfos.input_data_folder+"images_preprocessed_points.pkl","wb") as fp:
                pickle.dump(reformatted_dico, fp, pickle.HIGHEST_PROTOCOL)