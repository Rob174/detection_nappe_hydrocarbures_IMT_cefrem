from efficientnet_pytorch import EfficientNet


class ModelFactory:
    def __init__(self,model_name,num_classes=2):
        if model_name == "efficientnetv4":
            self.model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)
        elif model_name == "resnet":
            # TODO:
        elif model_name == "vgg16":
            # TODO:
        elif model_name == "deeplabv3":
            raise NotImplementedError()
        else:
            raise Exception("%s is not supported" % model_name)
    def __call__(self):
        return self.model