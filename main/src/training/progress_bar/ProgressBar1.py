"""Progress bar for a dataset with an unknwon number of samples and a known number of epochs (ClassificationPatch because it filters some unknwon patches)"""
from rich.progress import Progress, TextColumn

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar1(BaseClass, AbstractProgressBar):
    """Progress bar for a dataset with an unknwon number of samples and a known number of epochs (ClassificationPatch because it filters some unknwon patches)"""
    def __init__(self, num_epochs: int):
        super(ProgressBar1, self).__init__()
        self.columns.insert(8, TextColumn("[bold blue]num_processed_img: {task.fields[img_processed]:.4e}",
                                          justify="right"))
        self.columns.insert(9, "•")
        self._progress_bar = Progress(*self.columns)
        self._progress_bar_epochs = self._progress_bar.add_task("epochs", name="[red]Epochs", loss=0.,
                                                              total=num_epochs,
                                                              status=0, img_processed=0)

    def end_iteration(self, loss: float, epoch: int, img_processed: int, **kwargs):
        """Update ⚠⚠ epoch progress bar ⚠⚠ (as we cannot build an iteration progress bar for an unknown number of samples)

        Args:
            loss:float, loss of the model
            epoch: int, current epoch
            img_processed: int,number of images processed : we know the number of images provided at each train step and directly show the value on the epoch progress bar
            **kwargs:

        Returns:

        """
        self._progress_bar.update(self._progress_bar_epochs, advance=0, loss=loss, status=epoch,
                                 img_processed=img_processed)

    def end_epoch(self, loss: int, epoch: int, img_processed: int, **kwargs):
        """Update the epoch progress bar"""
        self._progress_bar.update(self._progress_bar_epochs, advance=1, loss=loss, status=epoch, img_processed=img_processed)

    @property
    def progress_bar(self) -> Progress:
        return self._progress_bar