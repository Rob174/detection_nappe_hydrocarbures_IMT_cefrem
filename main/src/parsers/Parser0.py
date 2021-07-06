import argparse

from main.src.data.Augmentation.Augmenters.enums import EnumAugmenter
from main.src.models.enums import EnumModels, EnumOptimizer
from main.src.param_savers.BaseClass import BaseClass
from main.src.data.classification.enums import EnumClassificationDataset
from main.src.data.enums import EnumUsage
from main.src.data.patch_creator.enums import *
from main.src.data.balance_classes.enums import *
from main.src.training.enums import EnumLoss


class Parser0(BaseClass):
    """
    Class managing possible arguments available to provide to the python main_script launched in the console

    Example 1: creating a parser and getting one argument

            >>> parser = Parser0() # initialize the parser (constructor)
            >>> arguments = parser() # parse arguments (by calling __call__ method)
            >>> arguments.dataset # example of access to one of the possible argument
            "classificationpatch1"
    """
    def __init__(self):
        self.attr_global_name = "parser"
        self.attr_name = self.__class__.__name__
        self.parser = argparse.ArgumentParser()
        self.args = {

                    '-no_security': {"name_or_flag":'no_security', "default":"false", "type":str,"help":"Indicate if you want to check if all python files have been commited before launching the training process true to disable the security"},
                    # Dataset
                    '-dataset':{"name_or_flag":'dataset',"default":"classificationpatch1","type":EnumClassificationDataset,"help":"Indicate the dataset used to constitue datasets","choices":list(EnumClassificationDataset)},
                    '-usage_type':{"name_or_flag":'usage_type',"default":"classification","type":EnumUsage,"help":"Indicate the source dataset used to constitue datasets","choices":list(EnumUsage)},
                    '-patch':{"name_or_flag":'patch',"default":"fixed_px","type":EnumPatchAlgorithm,"help":"Indicate the type of patch to create","choices":list(EnumPatchAlgorithm)},
                    '-patchExclPol': {"name_or_flag":'patch_exclude_policy', "default":"marginmorethan_1000", "type":str, "help":"Indicates the policy to exclude patches (especially patches containing margins)"},
                    '-grid_size':{"name_or_flag":'grid_size',"default":1000,"type":int,"help":"Indicate the grid size applied on the original image"},
                    '-in_size':{"name_or_flag":'input_size',"default":256,"type":int,"help":"Indicate the output size of the image obtained by resizing it after patches creation"},
                    '-bs':{"name_or_flag":'batch_size',"default":10,"type":int,"help":"Indique le nombre d'images par batch"},
                    '-balance':{"name_or_flag":'balance',"default":"balanceclasses1","type":EnumBalance,"help":"Indicate the policy to balance classes","choices":list(EnumBalance)},
                    # Augmentations
                    '-augmenter_img':{"name_or_flag":'augmenter_img',"default":"augmenter1","type":EnumAugmenter,"help":"Indicate which augmenter to use to apply transformations on source image","choices":list(EnumAugmenter)},
                    '-augmentations_img':{"name_or_flag":'augmentations_img',"default":"combinedRotResizeMir_10_0.25_4","type":str,"help":"Indicate the augmentations to apply to images in the order desired seprated by commas"},
                    '-augmenter_patch':{"name_or_flag":'augmenter_patch',"default":"noaugmenter","type":EnumAugmenter,"help":"Indicate which augmenter to use to use to apply transformations on patches","choices":list(EnumAugmenter)},
                    '-augmentations_patch':{"name_or_flag":'augmentations_patch',"default":"none","type":str,"help":"Indicate the augmentations to apply to patches in the order desired seprated by commas"},
                    '-augmentation_factor':{"name_or_flag":'augmentation_factor',"default":100,"type":int,"help":"The number of times that the source image is augmented"},
                    # Model
                    '-model':{"name_or_flag":'model',"default":"resnet18","type":EnumModels,"help":"To choose the network architecture used","choices":list(EnumModels)},
                    '-classes':{"name_or_flag":'classes',"default":"other,seep,spill","type":str,"help":"Indicate the class used for training separated by a comma"},
                    # Training
                    '-num_epochs':{"name_or_flag":'num_epochs',"default":100,"type":int,"help":"Number of epochs / repetitions of the training dataset"},
                    '-eval_step':{"name_or_flag":'eval_step',"default":10,"type":int,"help":"Number of training steps between two evaluation/validation steps"},
                    '-loss':{"name_or_flag":'loss_preference',"default":"binarycrossentropy","type":EnumLoss,"help":"Loss prefered for training","choices":list(EnumLoss)},

                    '-lr':{"name_or_flag":'lr',"default":1e-5,"type":float,"help":"Learning rate of the optimizer"},
                    '-eps':{"name_or_flag":'eps',"default":1e-7,"type":float,"help":"Epsilon of the optimizer if it is Adam"},
                    '-opti':{"name_or_flag":'optimizer',"default":"adam","type":EnumOptimizer,"help":"Optimizer algorithm","choices":list(EnumOptimizer)},

                    '-nbImg':{"name_or_flag":'nb_images',"default":-1,"type":int,"help":"Limit the number of images of the training dataset"},
                    '-debug':{"name_or_flag":'debug',"default":"true","type":str,"help":"Indicate if we want to save reference and predictions for each iteration"}
        }

    def __call__(self):
        return self.call()
    def call(self):
        """Parse arguments provided

        Returns: an object where we can access arguments provided as properties of the object (cf class examples)

        """
        for arg_name,dico_params in self.args.items():
            self.parser.add_argument(arg_name,**dico_params)
        return self.parser.parse_args()