from main.src.param_savers.BaseClass import BaseClass
import re
import numpy as np

class MetricsFactory(BaseClass):
    def __init__(self,*metrics_names):
        self.attr_list_metrics = {}
        self.functions_metrics = []
        for metric in metrics_names:
            metric = metric.lower()
            if re.match("^accuracy_classification-[0-9]\\.[0-9]+$",metric):
                self.attr_list_metrics[metric] = {"description":"Indicate the mean number of times accross batches where one probability of one image is equal to another with a margin of error of precision","tr_values":[],"valid_values":[]}
                precision = float(re.sub("^accuracy_classification-([0-9]\\.[0-9]+)$","\\1",metric))
                self.functions_metrics.append(lambda pred,true:np.mean(np.sum((np.abs(pred-true) < precision).astype(np.float),axis=1)))
            elif re.match("^mae$",metric):
                self.attr_list_metrics[metric] = {"description":"Indicate the mean mean absolute error accross batches","tr_values":[],"valid_values":[]}
                self.functions_metrics.append(lambda pred,true:np.mean(np.mean(np.abs(pred-true),axis=1)))
            else:
                raise NotImplementedError(f"{metric} has not been implemented")
    def __call__(self, prediction,true_value,dataset_type):
        prediction = prediction.detach().numpy()
        true_value = true_value.detach().numpy()
        for name,function in zip(self.attr_list_metrics.keys(),self.functions_metrics):
            self.attr_list_metrics[name][dataset_type+"_values"].append(function(prediction,true_value))