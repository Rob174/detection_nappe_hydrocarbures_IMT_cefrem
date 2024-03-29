"""Progress bar with a known number of iterations (case of ClassificationCache) (and a known number of epochs)"""

from rich.progress import TextColumn, Progress

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar0(BaseClass, AbstractProgressBar):
    """Progress bar with a known number of iterations (case of ClassificationCache) (and a known number of epochs)

    Args:
        length: int, length of the dataset
        num_epochs: int, number of epochs
    """
    def __init__(self, length: int, num_epochs: int):
        super(ProgressBar0, self).__init__()
        self._progress_bar = Progress(*self.columns)
        self._progress_bar_epochs = self._progress_bar.add_task("epochs", name="[red]Epochs", loss=0.,
                                                              total=num_epochs,
                                                              status=0)
        self._progress_bar_iterations = self._progress_bar.add_task("iterations", name="[blue]Global iterations", loss=0.,
                                                                  total=length * num_epochs,
                                                                  status=0)

    def end_iteration(self, loss: float, it_tr: int, tr_batch_size: int, **kwargs):
        """Update iteration progress bar

        Args:
            loss:float, loss of the model
            it_tr:it_tr, number of iterations of training elapsed
            tr_batch_size:int to update the number of samples provided to the model
            **kwargs:

        Returns:

        """
        self._progress_bar.update(self._progress_bar_iterations, advance=tr_batch_size, loss=loss, status=it_tr)

    def end_epoch(self, loss: int, epoch: int, **kwargs):
        """Update epoch progress bar

        Args:
            loss:float, loss of the model
            epoch, int
            **kwargs:

        Returns:
        """
        self._progress_bar.update(self._progress_bar_epochs, advance=1, loss=loss, status=epoch)

    @property
    def progress_bar(self) -> Progress:
        return self._progress_bar