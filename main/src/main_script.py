import json
import subprocess
import sys

from main.src.training.metrics_factory import MetricsFactory

path_root_project = r"C:\Users\robin\Documents\projets\detection_nappe_hydrocarbures_inria_cefrem"
sys.path.append(path_root_project)
sys.path.append(r"C:/Users/robin/Documents/projets/")
sys.path.append(path_root_project + f"\src")

from main.src.training.loss_factory import LossFactory
from main.src.training.optimizers_factory import OptimizersFactory
from main.FolderInfos import FolderInfos
from main.src.data.DatasetFactory import DatasetFactory
from main.src.parsers.Parser0 import Parser0
from main.src.models.ModelFactory import ModelFactory
from main.src.param_savers.saver0 import Saver0
import numpy as np

from torch.utils.data import random_split, DataLoader
import torch
from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TimeElapsedColumn, TimeRemainingColumn

if __name__ == "__main__":
    FolderInfos.init()
    dico_save_parameters = {"data": {}, "model": {}}
    arguments = Parser0()()
    saver = Saver0("")
    dataset = DatasetFactory(dataset_name=arguments.dataset,
                             usage_type=arguments.usage_type,
                             patch_creator=arguments.patch,
                             grid_size=arguments.grid_size,
                             input_size=arguments.input_size)
    dico_save_parameters["commit"] = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8").strip()
    dico_save_parameters["date"] = FolderInfos.id
    dico_save_parameters["data"]["dataset"] = saver(dataset)
    dico_save_parameters["data"]["prct_tr"] = 0.7
    dico_save_parameters["data"]["batch_size"] = arguments.batch_size
    dico_save_parameters["data"]["eval_step"] = arguments.eval_step
    length_full_dataset = len(dataset)
    tr_length = int(length_full_dataset * dico_save_parameters["data"]["prct_tr"])
    valid_length = length_full_dataset - tr_length
    [dataset_tr, dataset_valid] = random_split(dataset, lengths=[tr_length, valid_length],
                                               generator=torch.Generator().manual_seed(42))
    dataset_tr = DataLoader(dataset_tr, batch_size=dico_save_parameters["data"]["batch_size"], shuffle=True,
                            # shuffle the dataset at the beginning of each epoch
                            num_workers=0,  # num workers loading data into ram
                            prefetch_factor=2)  # there will be a total of 2 * num_workers samples prefetched across all workers
    dataset_valid = DataLoader(dataset_valid, batch_size=dico_save_parameters["data"]["batch_size"], shuffle=True,
                               # shuffle the dataset at the beginning of each epoch
                               num_workers=0,  # num workers loading data into ram
                               prefetch_factor=2)  # there will be a total of 2 * num_workers samples prefetched across all workers

    model = ModelFactory(model_name=arguments.model, num_classes=len(arguments.classes.split("_")))
    dico_save_parameters["model"] = saver(model)
    model = model()
    criterion = LossFactory(usage_type=arguments.usage_type, preference=arguments.loss_preference)
    optimizer = OptimizersFactory(model, name=arguments.optimizer,
                                  lr=arguments.lr, eps=arguments.eps)
    metrics = MetricsFactory("accuracy_classification-0.25","accuracy_classification-0.1","mae")
    dico_save_parameters["loss"] = saver(criterion)
    dico_save_parameters["optimizer"] = saver(optimizer)
    dico_save_parameters["metrics"] = saver(metrics)
    optimizer_pytorch = optimizer()
    criterion_pytorch = criterion()
    with open(FolderInfos.base_filename+"parameters.json","w") as fp:
        json.dump(dico_save_parameters,fp,indent=4)

    tr_length = len(dataset_tr)
    total_num_iterations = arguments.num_epochs * tr_length
    progress = Progress(
        TextColumn("{task.fields[name]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        TextColumn("[bold blue]status: {task.fields[status]}", justify="right"),
        "•",
        TextColumn("[bold blue]last_loss: {task.fields[loss]:.4f}", justify="right"),
        "•",
        TimeElapsedColumn(),
        "•",
        TimeRemainingColumn()
    )
    dico_save_parameters["training"] = {"tr_loss":[],"valid_loss":[]}
    dico_save_parameters["training"]["num_batches_tr"] = len(dataset_tr)
    print("start")
    with progress:
        epoch_progress = progress.add_task("epochs", name="[red]Epochs", loss=0., total=arguments.num_epochs,status=0)
        iterations_progress = progress.add_task("iterations", name="[bold blue]Total iterations", loss=0.,status=0,
                                                total=total_num_iterations)
        dataset_valid_iter = iter(dataset_valid)
        device = torch.device("cuda")
        model.to(device)

        for epoch in range(arguments.num_epochs):
            # print("epoch")

            for i, [input, output] in enumerate(dataset_tr):
                # TODO : add patch filtering here ; pb if no patches
                if dataset.attr_patch_creator.reject is True:
                    continue
                # print("step")
                # zero the parameter gradients
                optimizer_pytorch.zero_grad()

                # forward + backward + optimize
                prediction = model(input.to(device))
                loss = criterion_pytorch(prediction.to(device), output.to(device))
                loss.backward()
                optimizer_pytorch.step()
                current_loss = loss.item()

                dico_save_parameters["training"]["tr_loss"].append(float(current_loss))
                dico_save_parameters["training"]["last_it"] = i
                dico_save_parameters["training"]["last_epoch"] = epoch

                metrics(prediction.cpu(),output.cpu(),"tr")
                dico_save_parameters["metrics"] = saver(metrics)

                if i % arguments.eval_step == 0:
                    try:
                        input,output = next(dataset_valid_iter)
                    except StopIteration:
                        # StopIteration is thrown if dataset ends
                        # reinitialize data loader
                        dataset_valid_iter = iter(dataset_valid)
                        input,output = next(dataset_valid_iter)
                    prediction = model(input.to(device))
                    loss = criterion_pytorch(prediction.to(device), output.to(device))
                    current_loss = loss.item()
                    dico_save_parameters["training"]["valid_loss"].append(float(current_loss))
                    dico_save_parameters["data"]["dataset"] = saver(dataset)
                    metrics(prediction.cpu(),output.cpu(),"valid")
                    dico_save_parameters["metrics"] = saver(metrics)
                    with open(FolderInfos.base_filename+"parameters.json","r+") as fp:
                        json.dump(dico_save_parameters,fp,indent=4)
                    if loss < np.mean(dico_save_parameters["training"]["valid_loss"]) and i % 5 == 0:
                        torch.save(model.state_dict(),f"{FolderInfos.base_filename}_model_epoch-{epoch}_it-{i}.pt")



                progress.update(iterations_progress, advance=1, loss=current_loss,status=i)
            progress.update(epoch_progress, advance=1, loss=current_loss,status=epoch)

    print("end")