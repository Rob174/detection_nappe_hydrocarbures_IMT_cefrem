import rasterio
import numpy as np


def open_raster(pathImg):
    with rasterio.open(pathImg) as raster:
        image_array: np.ndarray = raster.read(1)
    return image_array

