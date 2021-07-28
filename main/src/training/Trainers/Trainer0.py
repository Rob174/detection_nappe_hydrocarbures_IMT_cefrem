"""Main trainer use for training purposes"""
import numpy as np
import torch
from typing import List

from main.src.data.DatasetFactory import DatasetFactory
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.IterationManager import IterationManager
from main.src.training.ObservableTrainer import ObservableTrainer
from main.src.training.TrValidSplit import trvalidsplit
from main.src.training.confusion_matrix.ConfusionMatrixCallback import ConfusionMatrixCallback
from main.src.training.early_stopping.AbstractEarlyStopping import AbstractEarlyStopping
from main.src.enums import *
from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
from main.src.training.progress_bar.ProgressBarFactory import ProgressBarFactory


class Trainer0(BaseClass,ObservableTrainer):
    """Main trainer use for training purposes

    Args:
        batch_size:
        num_epochs:
        tr_prct:
        dataset: DatasetFactory object to access data
        model:
        optimizer:
        loss:
        metrics:
        saver:
        eval_step:
        debug: str enum ("true" or "false") if "true", save prediction and annotation during training process. ⚠️ can slow down the training process
    """

    def __init__(self, batch_size, num_epochs, tr_prct,
                 dataset: DatasetFactory,
                 model: ModelFactory,
                 loss: AbstractLoss,
                 iteration_manager: IterationManager,
                 callbacks: List[AbstractCallback],
                 early_stopping: AbstractEarlyStopping,
                 eval_step: int,
                 debug: str="false"):
        """

        Args:
            batch_size: int, the training batch size: number of sample passed together to the attr_model
            num_epochs: int, number of complete processing of the attr_dataset by the attr_model
            tr_prct: float ∈ [0.,1.], percentage of the full attr_dataset dedicated to training
            dataset: DatasetFactory object to access data
            model: pytorch attr_model to train
            optimizer: optimizer factory
            loss: loss factory
            early_stopping: AbstractEarlyStopping to use
            eval_step: number of training batches between two validation steps
            debug: "false" or "true"  to store the predictions and the true values for the model in the json file
        """
        super(Trainer0, self).__init__(callbacks)
        self.attr_callbacks: List[AbstractCallback] = callbacks
        self.attr_debug = debug
        if debug == "true":
            self.attr_tr_vals_true = []
            self.attr_tr_vals_pred = []
        self.attr_iteration_manager = iteration_manager

        self.attr_dataset = dataset
        self.attr_loss: AbstractLoss = loss
        self.attr_early_stopping: AbstractEarlyStopping = early_stopping
        self.attr_progress = ProgressBarFactory(self.attr_dataset.len("tr"), num_epochs=num_epochs)
        self.attr_model = model

        # split the datasets into train and validation
        [dataset_tr, dataset_valid] = trvalidsplit(dataset)
        # create the dataloader classes to automatically generate the samples
        self.dataset_tr = dataset_tr  # there will be a total of 2 * num_workers samples prefetched across all workers
        self.dataset_valid = dataset_valid  # there will be a total of 2 * num_workers samples prefetched across all workers

        self.attr_global_name = "trainer"
        self.saver(self).save()

    def __call__(self):
        return self.call()
    def train(self,input_npy, output_npy, transformation_matrix, item,device):
        self.attr_model.model.train()
        self.attr_loss.zero_grad()

        # forward + backward + optimize
        input_gpu = torch.Tensor(input_npy).to(device)
        prediction_gpu = self.attr_model.model(input_gpu)
        del input_gpu
        output_gpu = torch.Tensor(output_npy).float().to(device)
        self.on_train_start(prediction_gpu, output_gpu)
        del output_gpu

        prediction_npy: np.ndarray = prediction_gpu.cpu().detach().numpy()
        self.on_train_end(prediction_npy, output_npy)

        # self.attr_progress.end_iteration(loss=self.attr_loss.attr_loss_values[EnumDataset.Train][-1], tr_batch_size=self.attr_tr_batch_size,
        #                                  attr_it_tr=attr_it_tr, img_processed=i,epoch=epoch)
    def valid(self,input_npy, output_npy, transformation_matrix, item,device):
        self.attr_model.model.eval()
        with torch.no_grad():
            input_gpu = torch.Tensor(input_npy).to(device)
            prediction = self.attr_model.model(input_gpu)
            del input_gpu
            prediction_npy: torch.Tensor = prediction.cpu().detach().numpy()
            self.on_valid_start(prediction_npy, output_npy)
            self.on_valid_end(prediction_npy, output_npy)

            # self.attr_loss(prediction, output_gpu, prediction_npy,output_npy, EnumDataset.Valid)
        # self.attr_metrics(prediction_npy, output_npy, EnumDataset.Valid)
    def call(self):
        """Train the attr_model"""
        with self.attr_progress:
            dataset_valid_iter = iter(self.dataset_valid)
            device = torch.device("cuda")
            self.attr_model.model.to(device)
            it_val = 0
            for epoch in range(self.attr_iteration_manager.attr_num_epochs):
                self.on_epoch_start(epoch)
                for it_tr, [input_npy, output_npy, transformation_matrix, item] in enumerate(self.dataset_tr):
                    self.train(input_npy, output_npy, transformation_matrix, item,device)
                    # Validation step
                    if it_tr % self.attr_iteration_manager.attr_eval_step == 0:
                        try:
                            input_npy, output_npy, transformation_matrix, item = next(dataset_valid_iter)
                        except StopIteration:
                            # reinitialize data loader
                            dataset_valid_iter = iter(self.dataset_valid)
                            input_npy, output_npy, transformation_matrix, item = next(dataset_valid_iter)
                        it_val += 1
                        self.attr_iteration_manager.on_valid_start(epoch, it_val)
                        self.valid(input_npy,output_npy,transformation_matrix,item,device)
                        self.attr_model_saver.save_model_if_required(self.attr_model, epoch, i)
                self.on_epoch_end(output_npy,prediction_npy,epoch,i)
                self.attr_progress.end_epoch(loss=self.attr_loss.attr_loss_values[EnumDataset.Train][-1], epoch=epoch, img_processed=i)
                self.attr_model_saver.save_model(self.attr_model, epoch, i)
                self.saver(self).save()
                if self.attr_early_stopping.stop_training():
                    break
            self.rgb_overlay(model=self.attr_model.model,
                             device = device)