import argparse
from main.src.param_savers.BaseClass import BaseClass

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

                    '-no_security': ['no_security', "false", str,
                                 "Indicate if you want to check if all python files have been commited before launching the training process {true to disable the security}"],
                    # Dataset
                    '-dataset':['dataset',"classificationpatch1",str,"Indicate the dataset used to constitue datasets {classificationpatch}"],
                    '-usage_type':['usage_type',"classification",str,"Indicate the source dataset used to constitue datasets {segmentation, classification}"],
                    '-patch':['patch',"fixed_px",str,"Indicate the type of patch to create {fixed_px}"],
                    '-patchExclPol': ['patch_exclude_policy', "marginmorethan_10000000000", str, "Indicates the policy to exclude patches (especially patches containing margins)"],
                    '-grid_size':['grid_size',1000,int,"Indicate the grid size applied on the original image"],
                    '-in_size':['input_size',256,int,"Indicate the output size of the image obtained by resizing it after patches creation"],
                    '-bs':['batch_size',10,int,"Indique le nombre d'images par batch"],
                    '-balance':['balance',"nobalance",str,"Indicate the policy to balance classes"],
                    '-balance_mg':['balance_margin',10,int,"Indicate the margin of supplementary image for one class"],
                    # Model
                    '-model':['model',"resnet18",str,"To choose the network architecture used {"],
                    '-classes':['classes',"seep,spill",str,"Indicate the class used for training separated by an underscore"],
                    # Training
                    '-num_epochs':['num_epochs',1,int,"Number of epochs / repetitions of the training dataset"],
                    '-eval_step':['eval_step',10,int,"Number of training steps between two evaluation/validation steps"],
                    '-loss':['loss_preference',"binarycrossentropy",str,"Loss prefered for training"],

                    '-lr':['lr',1e-5,float,"Learning rate of the optimizer"],
                    '-eps':['eps',1e-7,float,"Epsilon of the optimizer if it is Adam"],
                    '-opti':['optimizer',"adam",str,"Optimisateur"],

                    '-nbImg':['nb_images',-1,int,"Limit the number of images of the training dataset"],
                    '-debug':['debug',"false",str,"Indicate if we want to save reference and predictions for each iteration"],




                    '-nbEpochs':['nb_epochs',1,int,"Indique le nb de passage du dataset"],
                    '-augm':['augmentation',"f",str,"Indique le nb de passage du dataset"]}
    def __call__(self):
        return self.call()
    def call(self):
        """Parse arguments provided

        Returns: an object where we can access arguments provided as properties of the object (cf class examples)

        """
        for arg_name,[variable,default_val,type,description] in self.args.items():
            self.parser.add_argument(arg_name,dest=variable,default=default_val,type=type,help=description)
        return self.parser.parse_args()