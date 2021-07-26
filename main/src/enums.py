"""Contains enumerations EnumGitCheck"""

from enum import Enum


class EnumGitCheck(str, Enum):
    GITCHECK = "gitcheck"
    NOGITCHECK = "nogitcheck"

class EnumLoss(str, Enum):
    MulticlassnonExlusivCrossentropy = "multiclassnonexlusivcrossentropy"
    BinaryCrossentropy = "binarycrossentropy"
    MSE = "mse"


class EnumDataset(str, Enum):
    Train = "tr"
    Valid = "valid"

class EnumModels(str, Enum):
    Efficientnetv4 = "efficientnetv4"
    """Version b0"""
    Resnet18 = "resnet18"
    Vgg16 = "vgg16"
    Deeplabv3 = "deeplabv3"
    Resnet152 = "resnet152"


class EnumOptimizer(str, Enum):
    Adam = "adam"
    SGD = "sgd"


class EnumFreeze(str, Enum):
    AllExceptLastDense = "allexceptlastdense"
    NoFreeze = "nofreeze"

class EnumAugmenter(str, Enum):
    Augmenter0 = "augmenter0"
    """A step by step augmenter. See [Augmenter0](./Augmenter0.html)"""
    Augmenter1 = "augmenter1"
    """A one step augmenter. See [Augmenter1](./Augmenter1.html)"""
    NoAugmenter = "noaugmenter"
    """No augmentation applied"""

class EnumBalance(str, Enum):
    BalanceClasses2 = "balanceclasses2"
    """Exclude patches with classes other than the other category"""
    BalanceClasses1 = "balanceclasses1"
    """Exclude patches with only the other category"""
    NoBalance = "nobalance"
    """Does not filter pataches"""

class EnumLabelModifier(str, Enum):
    NoLabelModifier = "nolabelmodifier"
    """class other,seep,spill label"""
    LabelModifier1 = "labelmodifier1"
    """classes constructed to allow to choose classes to use"""
    LabelModifier2 = "labelmodifier2"
    """1 class constructed to allow to search if a combination of classes is present"""


class EnumClassPatchAdder(str, Enum):
    OtherClassPatchAdder = "other_class_patch_adder"
    NoClassPatchAdder = "no_class_patch_adder"

class EnumPatchAlgorithm(str, Enum):
    FixedPx = "fixed_px"
    """create the PatchCreator0 class which generate fixed px size patches. See [patch_creator0](./patch_creator0.html)"""


class EnumPatchExcludePolicy(str, Enum):
    MarginMoreThan = "marginmorethan"
    """Choose the option to exclude patches that have more than ... px at 0 in order to exclude patches with margins"""

class EnumUsage(str, Enum):
    Classification = "Generators"
    """attr_dataset to classify patches"""
    Segmentation = "Fabrics"
    """attr_dataset to segment an image"""


class EnumClasses(str, Enum):
    Other = "other"
    Seep = "seep"
    Spill = "spill"