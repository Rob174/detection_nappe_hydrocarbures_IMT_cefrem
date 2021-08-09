"""
We  have:
- 1 dataset hdf5 file with all polygons flattened
- 1 json file to split polygons from each other by writing their length
"""
import json
import numpy as np

from main.src.data.Datasets.DatasetOpener.JSONOpener import JSONOpener
from h5py import File

if __name__ == '__main__':
    infos = JSONOpener(
        r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_cache\filtered_cache_img_infos.json").dataset
    dico_infos = {}
    with File(
            r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\preprocessed_cache\world_earth_data\world_earth_data.hdf5",
            "r") as cache_src:
        with File(
                r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\filtered_cache\world_earth_data\world_earth_data.hdf5",
                "w") as cache_dst:
            for name, infos in infos.items():
                source_img = infos["source_img"]
                transformation_matrix = np.array((infos["transformation_matrix"]))
                polygons = np.array(cache_src[source_img])
                new_polygons = []
                for point_of_polygon in polygons:
                    new_polygons.append(transformation_matrix.dot([*point_of_polygon,1]))

                new_polygons = np.array(new_polygons)
                cache_dst.create_dataset(
                    name=name,
                    shape=new_polygons.shape,
                    dtype="f",
                    data=new_polygons
                )
