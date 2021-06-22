import time
from time import strftime, gmtime, localtime
import os
from platform import system
from main.src.param_savers.BaseClass import BaseClass


class FolderInfos(BaseClass):
    """Static class containing interesting folder of the project with a separator adapted for the operating system (windows or linux)
    We have to first initialize the class by calling the init method and then we can directly call the desired attribute

    Initialize some interesting pathes as static attributes.

    Args:
        custom_name: str, custom name to add in the folder of the current run
        subdir: str, subdir for datafolder
        test_without_data: bool, if False, the object automatically creates a folder for each run. If True this folder is not created.

    Static attributes:
        input_data_folder: str, data_in folder path
        data_folder: str,  data_out folder path
        base_folder: str, path to the newly created folder for the run
        base_filename: str, incomplete path to any new file in this new folder
        data_test: str, data_test folder path
        root_folder: str, path to the detection_nappe_hydrocarbures_IMT_cefrem folder
        id: str, uniq id generated with the datetime of the run
        separator: str, proper separator for the operating system (linux or windows)

    Example of usage:
    >>> FolderInfos.init(test_without_data=False)
    >>> FolderInfos.root_folder
    C:\\....\\detection_nappe_hydrocarbures_IMT_cefrem\\
    """
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
        """

        """
        """"""
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
            except Exception as e:
                print(e)
                print("waiting..... folder name already taken")
                time.sleep(4)
    @staticmethod
    def open_mode_suggestion(path):
        if os.path.exists(path) is True:
            return "r+"
        return "w"