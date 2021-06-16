import argparse
from main.src.param_savers.BaseClass import BaseClass

class Parser0(BaseClass):
    """Desiged for """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.args = {
                    # Dataset
                    '-dataset':['dataset',"sentinel1",str,"Indicate the source dataset used to constitue datasets {sentinel1}"],
                    '-usage_type':['usage_type',"classification",str,"Indicate the source dataset used to constitue datasets {segmentation, classification}"],
                    '-patch':['patch',"fixed_px",str,"Indicate the type of patch to create {fixed_px}"],
                    '-patchExclPol': ['patch_exclude_policy', "marginmorethan_1000", str, "Indicates the policy to exclude patches (especially patches containing margins)"],
                    '-grid_size':['grid_size',1000,int,"Indicate the grid size applied on the original image"],
                    '-in_size':['input_size',256,int,"Indicate the output size of the image obtained by resizing it after patches creation"],
                    '-bs':['batch_size',10,int,"Indique le nombre d'images par batch"],
                    # Model
                    '-model':['model',"resnet18",str,"To choose the network architecture used {"],
                    '-classes':['classes',"other_seep_spill",str,"Indicate the class used for training separated by an underscore"],

                    # Training
                    '-num_epochs':['num_epochs',1,int,"Number of epochs / repetitions of the training dataset"],
                    '-eval_step':['eval_step',10,int,"Number of training steps between two evaluation/validation steps"],
                    '-loss':['loss_preference',None,str,"Loss prefered for training"],

                    '-lr':['lr',1e-3,float,"Learning rate of the optimizer"],
                    '-eps':['eps',1e-7,float,"Epsilon of the optimizer if it is Adam"],
                    '-opti':['optimizer',"adam",str,"Optimisateur"],

                    '-nbImg':['nb_images',1080,int,"Limit the number of images of the training dataset"],




                    '-nbEpochs':['nb_epochs',1,int,"Indique le nb de passage du dataset"],
                    '-augm':['augmentation',"f",str,"Indique le nb de passage du dataset"]}

    def __call__(self, *args, **kwargs):
        for arg_name,[variable,default_val,type,description] in self.args.items():
            self.parser.add_argument(arg_name,dest=variable,default=default_val,type=type,help=description)
        return self.parser.parse_args()