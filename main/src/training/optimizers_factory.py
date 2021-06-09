
import torch.optim as optim

class OptimizersFactory:
    def __init__(self,model,name="adam",**params):
        self.attr_name = name
        if name == "adam":
            self.attr_params = {k:v for k,v in params.items() if k in ["lr","eps"]}
            self.optimizer = optim.Adam(model.parameters(),lr=params["lr"],eps=params["eps"])
        elif name == "sgd":
            self.attr_params = {k:v for k,v in params.items() if k in ["lr"]}
            self.optimizer = optim.SGD(model.parameters(),lr=params["lr"])
        else:
            raise NotImplementedError(f"{name} has not been implemented")
    def __call__(self):
        return self.optimizer