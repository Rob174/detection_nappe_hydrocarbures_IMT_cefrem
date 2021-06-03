from efficientnet_pytorch import EfficientNet


class ModelFactory:
    def __init__(self,model_name,num_classes=2):
        if model_name == "efficientnetv4":
            self.model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=num_classes)
        elif model_name == "deeplabv3":
            self.model =
    def __call__(self):
        return self.model