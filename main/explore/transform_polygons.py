"""
world_earth_data.json has the following structure:
{
    "name_img0": [
        [[x1,y1],...,[xn,yn]],
        [...],
        ..."
    ],
    ...
}
We want to map into a hdf5 file to reduce ram memory consumption and avoid opening all the polygons
We will have:
- 1 dataset hdf5 file with all polygons flattened
- 1 json file to split polygons from each other by writing their length
"""
import json
import numpy as np

from main.src.data.Datasets.DatasetOpener.JSONOpener import JSONOpener
from h5py import File
if __name__ == '__main__':
    data = JSONOpener(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\preprocessed_cache\world_earth_data\world_earth_data.json").dataset
    dico_infos = {}
    with File(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\preprocessed_cache\world_earth_data\world_earth_data.hdf5","w") as cache:
        for name,polygons in data.items():
            transformed_polygons = []
            dico_infos[name] = []
            for polygon in polygons:
                dico_infos[name].append(len(polygon))
                for point in polygon:
                    transformed_polygons.append(point)
            transformed_polygons = np.array(transformed_polygons,dtype=np.float32)
            cache.create_dataset(
                name=name,
                shape=transformed_polygons.shape,
                dtype="f",
                data=transformed_polygons
            )
    with open(r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_IMT_cefrem\data_in\preprocessed_cache\world_earth_data\world_earth_data_infos.json","w") as fp:
        json.dump(dico_infos,fp)
        