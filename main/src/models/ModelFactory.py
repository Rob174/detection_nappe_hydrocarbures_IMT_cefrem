from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from main.src.models.enums import *

from main.src.models.models.PretrainedModel import PretrainedModel
from main.src.param_savers.BaseClass import BaseClass


class ModelFactory(BaseClass):
    """Object creating desired model

    Args:
            model_name: EnumModels, model to create
            num_classes: int, number of output classes desired
            freeze: EnumFreeze
    """

    def __init__(self, model_name: EnumModels, num_classes: int = 2,
                 freeze: EnumFreeze = EnumFreeze.AllExceptLastDense):
        self.attr_global_name = "model"
        self.attr_freeze = freeze
        if model_name == EnumModels.Efficientnetv4:
            self.model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes)
        elif model_name == EnumModels.Resnet18:
            self.model = PretrainedModel(original_model=models.resnet18(pretrained=True),
                                         original_num_classes=1000,
                                         num_classes=num_classes, out_activation="sigmoid", freeze=freeze)
        elif model_name == EnumModels.Vgg16:
            self.model = PretrainedModel(original_model=models.vgg16(pretrained=True),
                                         original_num_classes=1000,
                                         num_classes=num_classes, freeze=freeze)
        elif model_name == EnumModels.Deeplabv3:
            raise NotImplementedError()
        else:
            raise Exception("%s is not supported" % model_name)
        self.attr_model_name = model_name
        self.attr_num_classes = num_classes

    def call(self):
        """Returns the pytorch model"""
        return self.model

    def __call__(self):
        return self.call()
