import torch

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback


class IterationManager(BaseClass,AbstractCallback):
    def __init__(self, valid_batch_size,train_batch_size,num_epochs,eval_step):
        super().__init__()
        self.attr_it_tr = 0
        self.attr_it_valid = 0
        self.attr_num_epochs = num_epochs
        self.attr_valid_size = valid_batch_size
        self.attr_tr_size = train_batch_size
        self.attr_last_epoch = 0
        self.attr_eval_step = eval_step
    def on_train_start(self, prediction_batch: torch.Tensor, true_batch: torch.Tensor):
        self.attr_it_tr += self.attr_tr_size
    def on_valid_start(self, prediction_batch, true_batch):
        self.attr_it_valid += self.attr_valid_size
    def on_epoch_start(self, epoch):
        self.attr_last_epoch = epoch
