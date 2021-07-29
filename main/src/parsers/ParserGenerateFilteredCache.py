import argparse

from main.src.enums import *
from main.src.param_savers.BaseClass import BaseClass


class ParserGenerateFilteredCache(BaseClass):
    """
    Class managing possible arguments available to provide to the python main_script launched in the console

    Example 1: creating a parser and getting one argument

            >>> parser = ParserGenerateFilteredCache() # initialize the parser (constructor)
            >>> arguments = parser() # parse arguments (by calling __call__ method)
            >>> arguments.attr_dataset # example of access to one of the possible argument
            "classificationpatch1"
    """

    def __init__(self):
        self.attr_global_name = "parser"
        self.attr_name = self.__class__.__name__
        self.parser = argparse.ArgumentParser()
        self.args = {

            '-no_security': {
                "dest": 'no_security',
                "default": EnumGitCheck.GITCHECK, "type": EnumGitCheck,
                "help": "Indicate if you want to check if all python files have been commited before launching the training process true to disable the security",
                "choices": list(EnumGitCheck)
            },
            # Dataset
            '-attr_dataset': {
                "dest": 'attr_dataset',
                "default": EnumLabelModifier.NoLabelModifier,
                "type": EnumLabelModifier,
                "help": "Indicate the attr_dataset used to constitue datasets",
                "choices": list(EnumLabelModifier)
            },
            '-usage_type': {
                "dest": 'usage_type',
                "default": EnumUsage.Classification,
                "type": EnumUsage,
                "help": "Indicate the source attr_dataset used to constitue datasets",
                "choices": list(EnumUsage)
            },
            '-patch': {
                "dest": 'patch',
                "default": EnumPatchAlgorithm.FixedPx,
                "type": EnumPatchAlgorithm,
                "help": "Indicate the type of patch to create",
                "choices": list(EnumPatchAlgorithm)
            },
            '-patchExclPol': {
                "dest": "patch_exclude_policy",
                "default": EnumPatchExcludePolicy.MarginMoreThan,
                "type": EnumPatchExcludePolicy,
                "help": "Indicates the policy to exclude patches (especially patches containing margins)",
                "choices": list(EnumPatchExcludePolicy)
            },
            '-patchExclThreshold': {
                "dest": "patch_exclude_policy_threshold",
                "default": 10,
                "type": int,
                "help": "The threshold used for margin more than algorithm"
            },
            '-grid_size': {
                "dest": 'grid_size',
                "default": 1000,
                "type": int,
                "help": "Indicate the grid size applied on the original image"
            },
            '-in_size': {
                "dest": 'input_size',
                "default": 256,
                "type": int,
                "help": "Indicate the output size of the image obtained by resizing it after patches creation"
            },
            '-bs': {
                "dest": 'batch_size',
                "default": 10,
                "type": int,
                "help": "Indique le nombre d'images par batch"
            },
            '-balance': {
                "dest": 'balance',
                "default": EnumBalance.BalanceClasses2,
                "type": EnumBalance,
                "help": "Indicate the policy to balance classes",
                "choices": list(EnumBalance)
            },
            '-classAdder': {
                "dest": 'other_class_adder',
                "default": EnumClassPatchAdder.NoClassPatchAdder,
                "type": EnumClassPatchAdder,
                "help": "Allow to add other patches in dataset (especially for the moment with only other class)"
            },
            '-interval': {
                "dest": 'interval',
                "default": 1,
                "type": int,
                "help": "Argument for OtherClassPatchAdder"
            },
            # Augmentations
            '-augmenter_img': {
                "dest": 'augmenter_img',
                "default": EnumAugmenter.NoAugmenter,
                "type": EnumAugmenter,
                "help": "Indicate which augmenter to use to apply transformations on source image",
                "choices": list(EnumAugmenter)
            },
            '-augmentations_img': {
                "dest": 'augmentations_img',
                "default": "combinedRotResizeMir_10_0.25_4",
                "type": str,
                "help": "Indicate the augmentations to apply to images in the order desired seprated by commas, available : combinedRotResizeMir_10_0.25_4"
            },

            '-augmentation_factor': {
                "dest": 'augmentation_factor',
                "default": 1,
                "type": int,
                "help": "The number of times that the source image is augmented"
            },
            # Model
            '-attr_model': {
                "dest": 'attr_model',
                "default": EnumModels.Resnet152,
                "type": EnumModels,
                "help": "To choose the network architecture used",
                "choices": list(EnumModels)
            },
            '-classes': {
                "dest": 'classes',
                "default": [EnumClasses.Seep, EnumClasses.Spill],
                "type": EnumClasses,
                "help": "Indicate the class used for training separated by a comma",
                "nargs": "+",
                "choices": list(EnumClasses)
            },
            '-freeze': {
                "dest": 'freeze',
                "default": EnumFreeze.NoFreeze,
                "type": EnumFreeze,
                "help": "Choose whcih layers to freeze in the original attr_model",
                "choices": list(EnumFreeze)
            },
            # Training
            '-num_epochs': {
                "dest": 'num_epochs',
                "default": 10,
                "type": int,
                "help": "Number of epochs / repetitions of the training attr_dataset"
            },
            '-eval_step': {
                "dest": 'eval_step',
                "default": 10,
                "type": int,
                "help": "Number of training steps between two evaluation/validation steps"
            },
            '-loss': {
                "dest": 'loss_preference',
                "default": "binarycrossentropy",
                "type": EnumLoss,
                "help": "Loss prefered for training",
                "choices": list(EnumLoss)
            },

            '-lr': {
                "dest": 'lr',
                "default": 1e-6,
                "type": float,
                "help": "Learning rate of the optimizer"
            },
            '-eps': {
                "dest": 'eps',
                "default": 1e-4,
                "type": float,
                "help": "Epsilon of the optimizer if it is Adam"
            },
            '-opti': {
                "dest": 'optimizer',
                "default": EnumOptimizer.Adam,
                "type": EnumOptimizer,
                "help": "Optimizer algorithm",
                "choices": list(EnumOptimizer)
            },

            '-nbImg': {
                "dest": 'nb_images',
                "default": -1,
                "type": int,
                "help": "Limit the number of images of the training attr_dataset"
            },
            '-debug': {
                "dest": 'debug',
                "default": "false",
                "type": str,
                "help": "Indicate if we want to save reference and predictions for each iteration"
            }
        }

    def __call__(self):
        return self.call()

    def call(self):
        """Parse arguments provided

        Returns: an object where we can access arguments provided as properties of the object (cf class examples)

        """
        for arg_name, dico_params in self.args.items():
            self.parser.add_argument(arg_name, **dico_params)
        self.parser.print_help()
        return self.parser.parse_args()
