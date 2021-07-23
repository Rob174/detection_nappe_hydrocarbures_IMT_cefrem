import numpy as np
import torch
from typing import List

from main.src.analysis.analysis.RGB_Overlay2 import RGB_Overlay2
from main.src.data.DatasetFactory import DatasetFactory
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.BaseClass import BaseClass
from main.src.training.AbstractCallback import AbstractCallback
from main.src.training.TrValidSplit import trvalidsplit
from main.src.training.confusion_matrix.ConfusionMatrixCallback import ConfusionMatrixCallback
from main.src.training.early_stopping.AbstractEarlyStopping import AbstractEarlyStopping
from main.src.enums import *
from main.src.training.metrics.loss_factory import LossFactory
from main.src.training.metrics.metrics_factory import MetricsFactory
from main.src.training.optimizers_factory import OptimizersFactory
from main.src.training.periodic_model_saver.AbstractModelSaver import AbstractModelSaver
from main.src.training.progress_bar.ProgressBarFactory import ProgressBarFactory


class TrainerTest(BaseClass):
    """Class managing the training process

    Args:
        batch_size: int, the training batch size: number of sample passed together to the attr_model
        num_epochs: int, number of complete processing of the attr_dataset by the attr_model
        tr_prct: float ∈ [0.,1.], percentage of the full attr_dataset dedicated to training
        dataset: DatasetFactory object to access data
        model: pytorch attr_model to train
        optimizer: optimizer factory
        loss: loss factory
        metrics: metrics factory
        saver: Saver0 object (see its documentation)
        eval_step: number of training batches between two validation steps
        debug: str enum ("true" or "false") if "true", save prediction and annotation during training process. ⚠️ can slow down the training process
    """

    def __init__(self, batch_size, num_epochs, tr_prct,
                 dataset: DatasetFactory,
                 model: ModelFactory,
                 optimizer: OptimizersFactory,
                 loss: LossFactory,
                 metrics: MetricsFactory,
                 early_stopping: AbstractEarlyStopping,
                 model_saver: AbstractModelSaver,
                 saver,
                 eval_step,
                 rgb_overlay: RGB_Overlay2,
                 debug="false"):

        self.attr_debug = debug
        if debug == "true":
            self.attr_tr_vals_true = []
            self.attr_tr_vals_pred = []
        self.attr_tr_batch_size = batch_size
        self.attr_tr_size = tr_prct
        self.attr_num_epochs = num_epochs
        self.attr_eval_step = eval_step
        self.attr_valid_batch_size = self.attr_tr_batch_size * self.attr_eval_step
        self.attr_prefetch_factor = 2  # only value possible with hdf5 files currently
        self.attr_save_step = self.attr_eval_step * 10

        self.attr_dataset = dataset
        self.attr_optimizer: OptimizersFactory = optimizer
        self.attr_loss: LossFactory = loss
        self.attr_metrics: MetricsFactory = metrics
        self.saver = saver
        self.attr_early_stopping: AbstractEarlyStopping = early_stopping
        self.attr_progress = ProgressBarFactory(self.attr_dataset.len("tr"), num_epochs=num_epochs)
        self.attr_model_saver = model_saver
        self.attr_model = model

        # split the datasets into train and validation
        [dataset_tr, dataset_valid] = trvalidsplit(dataset)
        # create the dataloader classes to automatically generate the samples
        self.dataset_tr = dataset_tr  # there will be a total of 2 * num_workers samples prefetched across all workers
        self.dataset_valid = dataset_valid  # there will be a total of 2 * num_workers samples prefetched across all workers

        self.attr_last_iter = -1
        self.attr_last_epoch = -1
        self.tr_batches = [[], []]
        self.valid_batches = [[], []]
        self.attr_global_name = "trainer"
        self.saver(self).save()
        self.attr_callbacks: List[AbstractCallback] = [
            ConfusionMatrixCallback(self.attr_dataset.attr_dataset.attr_label_modifier.get_final_class_mapping())
        ]
        self.rgb_overlay = rgb_overlay

    def add_to_batch_tr(self, input, output):
        """Add a sample to the current batch if it is not rejected and return the batch if it is full"""
        self.tr_batches[0].append(input)
        self.tr_batches[1].append(output)
        if len(self.tr_batches[0]) == self.attr_tr_batch_size:
            full_batch = (np.stack(self.tr_batches[0], axis=0), np.stack(self.tr_batches[1], axis=0))
            self.tr_batches = [[], []]
            return full_batch
        else:
            return None

    def add_to_batch_valid(self, input, output):
        """Add a sample to the current batch if it is not rejected and return the batch if it is full"""
        self.valid_batches[0].append(input)
        self.valid_batches[1].append(output)
        if len(self.valid_batches[0]) == self.attr_valid_batch_size:
            full_batch = (np.stack(self.valid_batches[0], axis=0), np.stack(self.valid_batches[1], axis=0))
            self.valid_batches = [[], []]
            return full_batch
        else:
            return None

    def __call__(self):
        return self.call()

    def call(self):
        """Train the attr_model"""
        with self.attr_progress:
            dataset_valid_iter = iter(self.dataset_valid)
            device = torch.device("cuda")
            self.attr_model.model.to(device)
            it_tr = 0
            it_val = 0
            for epoch in range(self.attr_num_epochs):
                for i, [input, output, transformation_matrix, item] in enumerate(self.dataset_tr):
                    opt_tr_batch = self.add_to_batch_tr(input, output)
                    if opt_tr_batch is not None:

                        # zero the parameter gradients
                        self.attr_optimizer.zero_grad()

                        # forward + backward + optimize

                        # Validation step
                        if it_tr % self.attr_eval_step == 0:
                            opt_valid_batch = None
                            while opt_valid_batch is None:
                                try:
                                    input, output, transformation_matrix, item = next(dataset_valid_iter)
                                except StopIteration:
                                    # reinitialize data loader
                                    dataset_valid_iter = iter(self.dataset_valid)
                                    input, output, transformation_matrix, item = next(dataset_valid_iter)
                                opt_valid_batch = self.add_to_batch_valid(input, output)
                            it_val += 1
                            input_npy, output_npy = opt_valid_batch
                            self.attr_model.model.eval()
                            with torch.no_grad():
                                input_gpu = torch.Tensor(input_npy).to(device)
                                prediction = self.attr_model.model(input_gpu)
                                del input_gpu
                                output_gpu = torch.Tensor(output_npy).float().to(device)
                                prediction_npy: torch.Tensor = prediction.cpu().detach().numpy()
                                for i in range(len(output_npy)):
                                    print(f"true : {output_npy[i]}, pred : {prediction_npy[i]}")
                                self.attr_loss(prediction, output_gpu, prediction_npy,output_npy, EnumDataset.Valid)
                            self.attr_metrics(prediction_npy, output_npy, EnumDataset.Valid)
                            for callback in self.attr_callbacks:
                                callback.on_valid_batch_end(prediction_npy,output_npy)
                        if it_val > 10:
                            break
                self.saver(self).save()
                if self.attr_early_stopping.stop_training():
                    break