import torch.nn as nn
import torch
from main.src.param_savers.BaseClass import BaseClass

class PretrainedModel(nn.Module,BaseClass):
    """Pretrained version of resnet with an additional layer to output the correct number of classes
    Done thanks to https://discuss.pytorch.org/t/changing-the-number-of-output-classes-of-fc-layer-of-vgg16/14346/3
    """
    def __init__(self,original_model,original_num_classes=1000,num_classes=3,out_activation="sigmoid"):
        super(PretrainedModel, self).__init__()
        self.net = original_model
        if out_activation == "sigmoid":
            self.activation = nn.Sigmoid()
        self.layer1 = nn.Linear(original_num_classes, num_classes) # convert from the original_num_classes classes from the original model pretrained on a dataset to num_classes classes
        for p in self.net.parameters():
            p.requires_grad = False # Freeze all weights of the original model

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.layer1(x1)
        y = self.activation(x2)
        return y