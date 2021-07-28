"""Progress bar for a dataset with an unknwon number of samples and a known number of epochs (ClassificationPatch because it filters some unknwon patches)"""
import numpy as np
from rich.progress import Progress, TextColumn

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.IterationManager import IterationManager
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar1(BaseClass, AbstractProgressBar):
    """Progress bar for a dataset with an unknwon number of samples and a known number of epochs (ClassificationPatch because it filters some unknwon patches)"""

    def __init__(self, iteration_manager: IterationManager, loss: AbstractLoss):
        super(ProgressBar1, self).__init__(iteration_manager, loss)
        self.columns.insert(8, TextColumn("[bold blue]num_processed_img: {task.fields[img_processed]:.4e}",
                                          justify="right"))
        self.columns.insert(9, "•")
        self._progress_bar = Progress(*self.columns)
        self._progress_bar_epochs = self._progress_bar.add_task(
            "epochs",
            name="[red]Epochs",
            loss=0.,
            total=self.iteration_manager.attr_num_epochs,
            status=0, img_processed=0
        )

    def on_train_end(self, prediction_batch: np.ndarray, true_batch: np.ndarray):
        """Update ⚠⚠ epoch progress bar ⚠⚠ (as we cannot build an iteration progress bar for an unknown number of samples)

        Args:
            loss:float, loss of the model
            **kwargs:

        Returns:

        """
        self._progress_bar.update(
            self._progress_bar_epochs,
            advance=0,
            loss=self.loss.get_last_tr_loss(),
            status=self.iteration_manager.attr_last_epoch,
            img_processed=self.iteration_manager.attr_tot_it_tr
        )

    def on_epoch_end(self):
        """Update the epoch progress bar"""
        self._progress_bar.update(self._progress_bar_epochs,
                                  advance=1,
                                  loss=self.loss.get_last_tr_loss(),
                                  status=self.iteration_manager.attr_last_epoch,
                                  img_processed=self.iteration_manager.attr_tot_it_tr
                                  )

    @property
    def progress_bar(self) -> Progress:
        return self._progress_bar
