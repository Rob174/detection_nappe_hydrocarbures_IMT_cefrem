from enum import Enum


class EnumModels(str, Enum):
    Efficientnetv4 = "efficientnetv4"
    """Version b0"""
    Resnet18 = "resnet18"
    Vgg16 = "vgg16"
    Deeplabv3 = "deeplabv3"


class EnumOptimizer(str, Enum):
    Adam = "adam"
    SGD = "sgd"


class EnumFreeze(str, Enum):
    AllExceptLastDense = "allexceptlastdense"
    NoFreeze = "nofreeze"
