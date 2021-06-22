from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from main.src.models.models.PretrainedModel import PretrainedModel
from main.src.param_savers.BaseClass import BaseClass


class ModelFactory(BaseClass):
    """Object creating desired model

    Args:
            model_name: str, name of the model to create. Currently supported:
            - "efficientnetv4"
            - "resnet18"
            - "vgg16"
            - "deeplabv3"
            num_classes: int, number of output classes desired
    """
    def __init__(self,model_name,num_classes=2):
        """


        """
        self.attr_global_name = "model"
        if model_name == "efficientnetv4":
            self.model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)
        elif model_name == "resnet18":
            self.model = PretrainedModel(original_model=models.resnet18(pretrained=True),
                                         original_num_classes=1000,
                                         num_classes=num_classes,out_activation="sigmoid")
        elif model_name == "vgg16":
            self.model = PretrainedModel(original_model=models.vgg16(pretrained=True),
                                         original_num_classes=1000,
                                         num_classes=num_classes)
        elif model_name == "deeplabv3":
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