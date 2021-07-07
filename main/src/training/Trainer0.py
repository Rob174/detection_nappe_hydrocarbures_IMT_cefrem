import json

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass
from torch.utils.data import random_split, DataLoader
import torch
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import numpy as np

from main.src.training.TrValidSplit import TrValidSplit, trvalidsplit


class Trainer0(BaseClass):
    """Class managing the training process

    Args:
        batch_size: int, the training batch size: number of sample passed together to the model
        num_epochs: int, number of complete processing of the dataset by the model
        tr_prct: float ∈ [0.,1.], percentage of the full dataset dedicated to training
        dataset: DatasetFactory object to access data
        model: pytorch model to train
        optimizer: pyrtoch optimizer to use
        loss: pytorch loss to use
        metrics: pytorch metrics to use
        saver: Saver0 object (see its documentation)
        eval_step: number of training batches between two validation steps
        debug: str enum ("true" or "false") if "true", save prediction and annotation during training process. ⚠️ can slow down the training process
    """
    def __init__(self ,batch_size,num_epochs,tr_prct,
                 dataset,
                 model,
                 optimizer,
                 loss,
                 metrics,
                 saver,
                 eval_step,debug="false"):

        self.attr_debug = debug
        if debug == "true":
            self.attr_tr_vals_true = []
            self.attr_tr_vals_pred = []
        self.attr_tr_batch_size = batch_size
        self.attr_tr_size = tr_prct
        self.attr_num_epochs = num_epochs
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.saver = saver
        self.attr_eval_step = eval_step
        self.attr_valid_batch_size = self.attr_tr_batch_size*self.attr_eval_step
        self.attr_prefetch_factor = 2 # only value possible with hdf5 files currently
        # split the datasets into train and validation
        [dataset_tr, dataset_valid] = trvalidsplit(dataset)
        # create the dataloader classes to automatically generate the samples
        self.dataset_tr = dataset_tr  # there will be a total of 2 * num_workers samples prefetched across all workers
        self.dataset_valid = dataset_valid  # there will be a total of 2 * num_workers samples prefetched across all workers
        self.model = model

        self.attr_tr_loss = []
        self.attr_valid_loss = []
        self.attr_last_iter = -1
        self.attr_last_epoch = -1
        self.tr_batches = [[],[]]
        self.valid_batches = [[],[]]
        self.progress_bar_creation()
        self.attr_global_name = "trainer"
        self.saver(self).save()
    def progress_bar_creation(self):
        """method to create the columns of the progressbar"""
        colums = [
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("[bold blue]status: {task.fields[status]}", justify="right"),
            "•",
            TextColumn("[bold blue]last_loss: {task.fields[loss]:.4e}", justify="right"),
            "•"
        ]
        self.length = self.dataset_tr.len()
        if self.length is None:
            colums.extend([
                TextColumn("[bold blue]num_processed_img: {task.fields[img_processed]:.4e}", justify="right"),
                "•"
            ])
        colums.extend([
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()
        ])
        self.progress = Progress(*colums)
    def add_to_batch_tr(self,input,output):
        """Add a sample to the current batch if it is not rejected and return the batch if it is full"""
        self.tr_batches[0].append(input)
        self.tr_batches[1].append(output)
        if len(self.tr_batches[0]) == self.attr_tr_batch_size:
            full_batch = (np.stack(self.tr_batches[0],axis=0),np.stack(self.tr_batches[1],axis=0))
            self.tr_batches = [[],[]]
            return full_batch
        else:
            return None
    def add_to_batch_valid(self,input,output):
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
        """Train the model"""
        with self.progress:

            if self.length is not None:
                global_iteration_progress = self.progress.add_task("iterations", name="[blue]Global iterations", loss=0., total=self.length,
                                               status=0)
                epoch_progress = self.progress.add_task("epochs", name="[red]Epochs", loss=0.,
                                                        total=self.attr_num_epochs,
                                                        status=0)
            else:

                epoch_progress = self.progress.add_task("epochs", name="[red]Epochs", loss=0.,
                                                        total=self.attr_num_epochs,
                                                        status=0,img_processed=0)
            dataset_valid_iter = iter(self.dataset_valid)
            device = torch.device("cuda")
            self.model.to(device)
            current_loss = -1
            it_tr = 0
            it_val = 0
            opt_valid_batch = None
            for epoch in range(self.attr_num_epochs):
                for i, [input, output,transformation_matrix,item] in enumerate(self.dataset_tr):
                    opt_tr_batch = self.add_to_batch_tr(input,output)
                    if opt_tr_batch is not None:
                        it_tr += 1
                        input,output = opt_tr_batch
                        input = torch.Tensor(input)
                        output = torch.Tensor(output)
                        # zero the parameter gradients
                        self.optimizer.zero_grad()

                        # forward + backward + optimize
                        prediction = self.model(input.to(device))
                        loss = self.loss(prediction.float().to(device), output.float().to(device))
                        loss.backward()
                        self.optimizer.step()
                        current_loss = loss.item()

                        self.attr_tr_loss.append(float(current_loss))
                        self.attr_last_iter = i
                        self.attr_last_epoch = epoch
                        prediction: torch.Tensor = prediction.cpu()
                        output: torch.Tensor = output.cpu()
                        if self.attr_debug == "true":
                            self.attr_tr_vals_true.append(output.detach().numpy().tolist())
                            self.attr_tr_vals_pred.append(prediction.detach().numpy().tolist())
                        self.metrics(prediction, output, "tr")
                        self.saver(self.metrics)

                        if self.length is not None:
                            self.progress.update(global_iteration_progress, advance=1, loss=current_loss, status=it_tr)
                        if self.length is None:
                            self.progress.update(epoch_progress, advance=0, loss=current_loss, status=epoch,img_processed=i)
                        # Validation step
                        if it_tr % self.attr_eval_step == 0:
                            opt_valid_batch = None
                            while opt_valid_batch is None:
                                try:
                                    input, output,transformation_matrix,item = next(dataset_valid_iter)
                                except StopIteration:
                                    # StopIteration is thrown if dataset ends
                                    # reinitialize data loader
                                    dataset_valid_iter = iter(self.dataset_valid)
                                    input, output,transformation_matrix,item = next(dataset_valid_iter)
                                opt_valid_batch = self.add_to_batch_valid(input,output)
                                it_val += 1
                            input,output = opt_valid_batch
                            input = torch.Tensor(input)
                            output = torch.Tensor(output)
                            prediction = self.model(input.to(device))
                            loss = self.loss(prediction.float().to(device), output.float().to(device))
                            current_loss = loss.item()
                            self.attr_valid_loss.append(float(current_loss))
                            self.saver(self.dataset)
                            self.metrics(prediction.cpu(), output.cpu(), "valid")
                            self.saver(self.metrics)
                            self.saver(self).save()

                            if loss < np.mean(self.attr_valid_loss) and it_tr % 10 == 0:
                                torch.save(self.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{i}.pt")

                if self.length is not None:
                    self.progress.update(epoch_progress, advance=1, loss=current_loss, status=epoch)
                else:
                    self.progress.update(epoch_progress, advance=1, loss=current_loss, status=epoch,img_processed=i)
                torch.save(self.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{i}.pt")
                self.saver(self.dataset)
                self.saver(self.metrics)
                self.saver(self).save()