from main.src.data.Datasets.PointExtractor.AbstractPointExtractor import AbstractPointExtractor
from main.src.data.preprocessing.point_shapes_to_file import EnumShapeCategories


class PointExtractorSingleCategoryDict(AbstractPointExtractor):
    def extract(self,data):
        return data
