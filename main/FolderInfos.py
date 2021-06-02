import time
from time import strftime, gmtime, localtime
import os
from platform import system


class FolderInfos:
    input_data_folder = None
    data_folder = None
    base_folder = None
    base_filename = None
    data_test = None
    root_folder = None
    id = None
    separator = "/"

    @staticmethod
    def init(custom_name="",subdir="",test_without_data=False):
        if system() != "Linux":
            FolderInfos.separator = "\\"
        while True:
            FolderInfos.id = strftime("%Y-%m-%d_%Hh%Mmin%Ss", localtime())
            FolderInfos.root_folder = FolderInfos.separator.join(os.path.realpath(__file__).split(FolderInfos.separator)[:-2]) + FolderInfos.separator
            FolderInfos.data_folder = FolderInfos.separator.join(os.path.realpath(__file__).split(FolderInfos.separator)[:-2] + ["data_out"+FolderInfos.separator])
            FolderInfos.input_data_folder = FolderInfos.separator.join(os.path.realpath(__file__).split(FolderInfos.separator)[:-2] + ["data_in"]) + FolderInfos.separator
            if subdir != "":
                FolderInfos.data_folder += subdir +FolderInfos.separator if subdir[-1] != FolderInfos.separator else subdir
            FolderInfos.base_folder = FolderInfos.data_folder + FolderInfos.id + "_"+custom_name+FolderInfos.separator
            FolderInfos.base_filename = FolderInfos.base_folder + FolderInfos.id+"_"+custom_name
            FolderInfos.data_test = FolderInfos.separator.join(os.path.realpath(__file__).split(FolderInfos.separator)[:-2] + ["data_test"]) + FolderInfos.separator
            try:
                if test_without_data is False:
                    os.mkdir(FolderInfos.base_folder)
                break
            except:
                print("waiting..... folder name already taken")
                time.sleep(4)