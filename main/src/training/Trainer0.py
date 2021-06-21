import json

from main.FolderInfos import FolderInfos
from main.src.param_savers.BaseClass import BaseClass
from torch.utils.data import random_split, DataLoader
import torch
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.console import Console
import numpy as np


class Trainer0(BaseClass):
    def __init__(self ,batch_size,num_epochs,tr_prct,
                 dataset,
                 model,
                 optimizer,
                 loss,
                 metrics,
                 saver,
                 eval_step):
        self.attr_tr_batch_size = batch_size
        self.attr_tr_size = tr_prct
        self.attr_num_epochs = num_epochs
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        length_full_dataset = len(dataset)
        self.tr_length = int(length_full_dataset * tr_prct)
        self.saver = saver
        self.attr_eval_step = eval_step
        self.attr_valid_batch_size = self.attr_tr_batch_size*self.attr_eval_step
        valid_length = length_full_dataset - self.tr_length
        self.attr_prefetch_factor = 2
        [dataset_tr, dataset_valid] = random_split(dataset, lengths=[self.tr_length, valid_length],
                                                   generator=torch.Generator().manual_seed(42))
        self.dataset_tr = DataLoader(dataset_tr, batch_size=1, shuffle=True,
                                # shuffle the dataset at the beginning of each epoch
                                num_workers=0,  # num workers loading data into ram
                                prefetch_factor=self.attr_prefetch_factor)  # there will be a total of 2 * num_workers samples prefetched across all workers
        self.dataset_valid = DataLoader(dataset_valid, batch_size=1, shuffle=True,
                                   # shuffle the dataset at the beginning of each epoch
                                   num_workers=0,  # num workers loading data into ram
                                   prefetch_factor=self.attr_prefetch_factor)  # there will be a total of 2 * num_workers samples prefetched across all workers
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
        self.progress = Progress(
            TextColumn("{task.fields[name]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%",
            "•",
            TextColumn("[bold blue]status: {task.fields[status]}", justify="right"),
            "•",
            TextColumn("[bold blue]last_loss: {task.fields[loss]:.4e}", justify="right"),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn()
        )
    def add_to_batch_tr(self,input,output,reject):
        if reject is True:
            return None
        self.tr_batches[0].append(input)
        self.tr_batches[1].append(output)
        if len(self.tr_batches[0]) == self.attr_tr_batch_size:
            full_batch = (np.stack(self.tr_batches[0],axis=0),np.stack(self.tr_batches[1],axis=0))
            self.tr_batches = [[],[]]
            return full_batch
        else:
            return None
    def add_to_batch_valid(self,input,output,reject):
        if reject is True:
            return None
        self.valid_batches[0].append(input)
        self.valid_batches[1].append(output)
        if len(self.valid_batches[0]) == self.attr_valid_batch_size:
            full_batch = (np.stack(self.valid_batches[0], axis=0), np.stack(self.valid_batches[1], axis=0))
            self.valid_batches = [[], []]
            return full_batch
        else:
            return None
    def __call__(self):
        with self.progress:
            epoch_progress = self.progress.add_task("epochs", name="[red]Epochs", loss=0., total=self.attr_num_epochs,
                                               status=0)
            iterations_progress = self.progress.add_task("iterations", name="[bold blue]Total iterations", loss=0., status=0,
                                                    total=self.attr_num_epochs * self.tr_length)
            dataset_valid_iter = iter(self.dataset_valid)
            device = torch.device("cuda")
            self.model.to(device)
            current_loss = -1
            import time
            for epoch in range(self.attr_num_epochs):
                # print("epoch")

                for i, [input, output,reject] in enumerate(self.dataset_tr):

                    opt_tr_batch = self.add_to_batch_tr(input,output,reject)
                    if opt_tr_batch is not None:
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

                        self.metrics(prediction.cpu(), output.cpu(), "tr")
                        self.saver(self.metrics)

                    try:
                        input, output,reject = next(dataset_valid_iter)
                    except StopIteration:
                        # StopIteration is thrown if dataset ends
                        # reinitialize data loader
                        dataset_valid_iter = iter(self.dataset_valid)
                        input, output,reject = next(dataset_valid_iter)
                    opt_valid_batch = self.add_to_batch_valid(input,output,reject)
                    if opt_valid_batch is not None:
                        prediction = self.model(input.to(device))
                        loss = self.loss(prediction.float().to(device), output.float().to(device))
                        current_loss = loss.item()
                        self.attr_valid_loss.append(float(current_loss))
                        self.saver(self.dataset)
                        self.metrics(prediction.cpu(), output.cpu(), "valid")
                        self.saver(self.metrics)
                        self.saver(self).save()

                        if loss < np.mean(self.attr_valid_loss) or i % 1000 == 0:
                            torch.save(self.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{i}.pt")

                    self.progress.update(iterations_progress, advance=1, loss=current_loss, status=i)
                self.progress.update(epoch_progress, advance=1, loss=current_loss, status=epoch)
            torch.save(self.model.state_dict(), f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{i}.pt")