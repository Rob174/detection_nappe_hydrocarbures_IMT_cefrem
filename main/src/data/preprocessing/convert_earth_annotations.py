import json

from main.FolderInfos import FolderInfos as FI
from main.src.data.Datasets.ColorExtractor.DicoColorExtractor import DicoColorExtractor
from main.src.data.Datasets.PointDataset import PointDataset
from main.src.data.Datasets.PointExtractor.PointExtractorMultiCategoryDict import PointExtractorMultiCategoryDict
from main.src.data.TwoWayDict import TwoWayDict

import numpy as np

if __name__ == '__main__':
    FI.init(test_without_data=True)
    mapping = TwoWayDict(
            {  # Formatted in the following way: src_index in cache, name, the position encode destination index
                0: "other",
                1: "seep",
                2: "spill"
            }
        )

    dataset = PointDataset(
        path_pkl=FI.input_data_folder+"preprocessed_cache"+FI.separator+"world_earth_data.pkl",
        mapping=mapping,
        color_extractor=DicoColorExtractor(mapping),
        point_extractor=PointExtractorMultiCategoryDict()
    )
    with open(FI.input_data_folder+"filtered_cache"+FI.separator+"filtered_cache_img_infos.json","r") as fp:
        dico_info_cache = json.load(fp)
    for name,dico_infos in dico_info_cache.items():
        source_img = dico_infos["source_img"]
        transformation_matrix = np.array(dico_infos["transformation_matrix"])
        polygons = dataset.get(source_img)
        new_polygons = []
        for polygon in polygons:
            new_polygon = []
            for point in polygon:
                new_polygon.append(transformation_matrix.dot(np.array([*point,1])).tolist()[:-1])