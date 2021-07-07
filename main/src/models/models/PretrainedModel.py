import torch.nn as nn
import torch

from main.src.models.enums import EnumFreeze
from main.src.param_savers.BaseClass import BaseClass

class PretrainedModel(nn.Module,BaseClass):
    """Pretrained version of resnet with an additional layer to output the correct number of classes
    Done thanks to https://discuss.pytorch.org/t/changing-the-number-of-output-classes-of-fc-layer-of-vgg16/14346/3


    Args:
        original_model: the original pretrained model ⚠️ with weights loaded
        original_num_classes: int, the number of classes given as output by the original pretrained model
        num_classes: int, number of desired number of classes
        out_activation: str, activation to add at the end of the network. Currently supported:
        - sigmoid
        freeze: EnumFreeze
    """
    def __init__(self,original_model,original_num_classes=1000,num_classes=3,out_activation="sigmoid", freeze: EnumFreeze = EnumFreeze.AllExceptLastDense):
        super(PretrainedModel, self).__init__()
        self.net = original_model
        if out_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        self.layer1 = nn.Linear(original_num_classes, num_classes) # convert from the original_num_classes classes from the original model pretrained on a dataset to num_classes classes

        for p in self.net.parameters():
            if freeze == EnumFreeze.AllExceptLastDense:
                p.requires_grad = False # Freeze all weights of the original model
            elif freeze == EnumFreeze.NoFreeze:
                p.requires_grad = True

    def forward(self, x):
        """ Forward pass on the network with the desired output number of classes and activation function

        Args:
            x: input for the network

        Returns:
            prediction of the network

        """
        x1 = self.net(x)
        x2 = self.layer1(x1)
        y = self.activation(x2)
        return y