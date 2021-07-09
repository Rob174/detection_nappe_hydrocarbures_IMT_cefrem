from rich.progress import TextColumn, Progress

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar0(BaseClass, AbstractProgressBar):
    def __init__(self, length: int, num_epochs: int):
        super(ProgressBar0, self).__init__()
        self.columns.insert(8, TextColumn("[bold blue]num_processed_img: {task.fields[img_processed]:.4e}",
                                          justify="right"))
        self.columns.insert(9, "•")
        columns = self.columns
        self._progress_bar = Progress(*columns)
        self._progress_bar_epochs = self._progress_bar.add_task("epochs", name="[red]Epochs", loss=0.,
                                                              total=num_epochs,
                                                              status=0)
        self._progress_bar_iterations = self._progress_bar.add_task("iterations", name="[blue]Global iterations", loss=0.,
                                                                  total=length * num_epochs,
                                                                  status=0)

    def end_iteration(self, loss: float, it_tr: int, tr_batch_size: int, **kwargs):
        self._progress_bar.update(self._progress_bar_iterations, advance=tr_batch_size, loss=loss, status=it_tr)

    def end_epoch(self, loss: int, epoch: int, **kwargs):
        self._progress_bar.update(self._progress_bar_epochs, advance=1, loss=loss, status=epoch)
