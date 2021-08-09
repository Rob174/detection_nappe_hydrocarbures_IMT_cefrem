import json

from typing import List, Dict, Union

from main.FolderInfos import FolderInfos as FI
from main.src.data.Datasets.ColorExtractor.DicoColorExtractor import DicoColorExtractor
from main.src.data.Datasets.Fabrics.FabricPreprocessedCache import FabricPreprocessedCache
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.Datasets.PointExtractor.PointExtractorMultiCategoryDict import PointExtractorMultiCategoryDict
from main.src.data.TwoWayDict import TwoWayDict

import numpy as np

from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories

if __name__ == '__main__':
    FI.init(test_without_data=True)
    images, annotations, informations = FabricPreprocessedCache()()
    with open(FI.input_data_folder + "filtered_cache" + FI.separator + "filtered_cache_img_infos.json", "r") as fp:
        dico_info_cache = json.load(fp)
    for name, dico_infos in dico_info_cache.items():
        source_img = dico_infos["source_img"]
        transformation_matrix = np.array(dico_infos["transformation_matrix"])
        polygons: List[Dict[EnumShapeCategories, Union[str, List]]] = annotations.get(source_img)
        new_polygons = []
        for polygon in polygons:
            new_polygon = []
            points = polygon[EnumShapeCategories.Points]
            category = polygon[EnumShapeCategories.Label]
            for point in points:
                new_polygon.append(transformation_matrix.dot(np.array([*point, 1])).tolist()[:-1])
            if np.max(new_polygons) > 0:
                new_polygons.append({EnumShapeCategories.Label: category, EnumShapeCategories.Points: new_polygon})
    with open(FI.input_data_folder + "filtered_cache"+FI.separator+"filtered_cache_world_earth_data.json", "w") as fp:
        json.dump(new_polygons, fp)
