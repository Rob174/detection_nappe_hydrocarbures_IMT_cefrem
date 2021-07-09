import numpy as np
import rasterio


def open_raster(pathImg):
    with rasterio.open(pathImg) as raster:
        image_array: np.ndarray = raster.read(1)
    return image_array, np.array(raster.transform).reshape(3, 3)
