from typing import List

import rasterio
from rich.console import Console
from rich.table import Table

from main.FolderInfos import FolderInfos
import os
import re
import numpy as np
import time

if __name__ == "__main__":
    # Get file names ....
    FolderInfos.init(test_without_data=True)
    files = [FolderInfos.data_test + f for f in os.listdir(FolderInfos.data_test)] # full path of files
    dico_by_extensions = {} # will store all files accessible their extension then name to get the full path
    for sf, f in zip(os.listdir(FolderInfos.data_test), files):
        # Extract the name, starting from the date of the file without the extension
        name = re.sub("^([A-Za-z0-9_]+[A-Za-z_]{,2})(20[0-9_A-Z]+)(\\.[a-z]+)$", "\\2", sf)
        # Get the extension of the file
        ext = sf.split(".")[-1]
        if ext not in dico_by_extensions: # if the extension is not a key of the dictionnary
            dico_by_extensions[ext] = {} # we create it and create the inner dict which will store all the pairs name, fullpath
        dico_by_extensions[ext][name] = f # store the pair name, fullpath
    # Measures...
    list_imgs_ids = ["027481_0319CB_0EB7", "016505_01F10F_CE84", "016753_01F88A_4864"]
    dico_times = {k: {"values": [], "shape": None, "mem_size": None} for k in list_imgs_ids}
    for i in range(10):
        print(f"Step {i}")
        for [[name, path], uniq_id] in zip(dico_by_extensions["img"].items(), list_imgs_ids):
            initial_time = time.time_ns()
            with rasterio.open(path) as file_object:
                dataset = file_object.read(1)
            dico_times[uniq_id]["values"].append(time.time_ns() - initial_time) # save the excution time in ns
            dico_times[uniq_id]["shape"] = dataset.shape
            dico_times[uniq_id]["mem_size"] = dataset.nbytes# save the number of bytes take by the array representer the raster image
    print("We have the following access time for each image:")
    console = Console(color_system="windows") # Declare the object console which will manage improved logs
    table = Table(show_header=True, header_style="bold magenta") # Create the table and then add the columns names
    table.add_column("Name")
    table.add_column("Access time (ms)")
    table.add_column("Shape of the image")
    table.add_column("Memory size (MB)")
    for name_img, values in dico_times.items():
        table.add_row(
            f"{name_img} avg time", str(np.mean(values["values"]) * 1e-6), str(values["shape"]),
            str(values["mem_size"] * 1e-6) # Add row with values (str: mandatory) in the same order as their corresponding column
        )
    all_times: List[List[float]] = [v["values"] for v in dico_times.values()]
    table.add_row(
        f"Global avg time", str(np.mean(np.concatenate(all_times, axis=0) * 1e-6)), "-"
    )
    console.print(table) # show the table

    import h5py

    images_hdf5 = h5py.File(f"{FolderInfos.input_data_folder}test_tmp_images.hdf5", "w") # create the hdf5 file

    # Measures...
    list_imgs_ids = ["027481_0319CB_0EB7", "016505_01F10F_CE84", "016753_01F88A_4864"]
    dico_times = {k: {"valuesRead": [], "valuesWrite": [], "shape": None, "mem_size": None} for k in list_imgs_ids}
    for [[name, path], uniq_id] in zip(dico_by_extensions["img"].items(), list_imgs_ids):
        ## Open the raster
        with rasterio.open(path) as raster:  ## (NB: with keyword allows to manage files (properly open and close them)
            image_array: np.ndarray = raster.read(1)  # Get the image array
            dico_times[uniq_id]["shape"] = image_array.shape
            dico_times[uniq_id]["mem_size"] = image_array.nbytes

        initial_time = time.time_ns()
        images_hdf5.create_dataset(uniq_id, shape=image_array.shape, dtype='f',
                                   data=image_array)
        dico_times[uniq_id]["valuesWrite"].append(time.time_ns() - initial_time)

    images_hdf5.flush()
    images_hdf5.close()
    images_hdf5 = h5py.File(f"{FolderInfos.input_data_folder}test_tmp_images.hdf5", "r")

    for [[name, path], uniq_id] in zip(dico_by_extensions["img"].items(), list_imgs_ids):
        initial_time = time.time_ns()
        array = images_hdf5[uniq_id]
        a = 1
        dico_times[uniq_id]["valuesRead"].append(time.time_ns() - initial_time)
        array2 = np.copy(array) * 2

    console = Console(color_system="windows")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Name")
    table.add_column("Access time write (ms)")
    table.add_column("Access time read (ns)")
    table.add_column("Shape of the image")
    table.add_column("Memory size (MB)")
    for name_img, values in dico_times.items():
        table.add_row(
            f"{name_img} avg time", str(np.mean(values["valuesWrite"]) * 1e-6), str(np.mean(values["valuesRead"])),
            str(values["shape"]), str(values["mem_size"] * 1e-6)
        )
    all_times_read = [v["valuesRead"] for v in dico_times.values()]
    all_times_write = [v["valuesWrite"] for v in dico_times.values()]
    table.add_row(
        f"Global avg time", str(np.mean(np.concatenate(all_times_write, axis=0) * 1e-6)),
        str(np.mean(np.concatenate(all_times_read, axis=0))), "-"
    )
    console.print(table)