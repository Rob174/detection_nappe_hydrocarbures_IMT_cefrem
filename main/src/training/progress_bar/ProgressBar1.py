from rich.progress import Progress

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar1(BaseClass, AbstractProgressBar):
    def __init__(self, num_epochs: int):
        super(ProgressBar1, self).__init__()
        self.progress_bar = Progress(*self.columns)
        self.progress_bar_epochs = self.progress_bar.add_task("epochs", name="[red]Epochs", loss=0.,
                                                              total=num_epochs,
                                                              status=0, img_processed=0)

    def end_iteration(self, loss: float, epoch: int, img_processed: int, **kwargs):
        self.progress_bar.update(self.progress_bar_iterations, advance=0, loss=loss, status=epoch,
                                 img_processed=img_processed)

    def end_epoch(self, loss: int, epoch: int, img_processed: int, **kwargs):
        self.progress.update(self.progress_bar_epochs, advance=1, loss=loss, status=epoch, img_processed=img_processed)
