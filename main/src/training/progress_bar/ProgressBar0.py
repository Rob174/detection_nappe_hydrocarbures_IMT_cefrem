"""Progress bar with a known number of iterations (case of ClassificationCache) (and a known number of epochs)"""

import numpy as np
from rich.progress import Progress

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.IterationManager import IterationManager
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
from main.src.training.progress_bar.AbstractProgressBar import AbstractProgressBar


class ProgressBar0(BaseClass, AbstractProgressBar):
    """Progress bar with a known number of iterations (case of ClassificationCache) (and a known number of epochs)

    Args:
        length: int, length of the dataset
        num_epochs: int, number of epochs
    """

    def __init__(self, length: int, iteration_manager: IterationManager, loss: AbstractLoss):
        super(ProgressBar0, self).__init__(iteration_manager, loss)
        self._progress_bar = Progress(*self.columns)
        self._progress_bar_epochs = self._progress_bar.add_task(
            "epochs",
            name="[red]Epochs",
            loss=0.,
            total=iteration_manager.attr_num_epochs,
            status=0
        )
        self._progress_bar_iterations = self._progress_bar.add_task(
            "iterations",
            name="[blue]Global iterations",
            loss=0.,
            total=length * iteration_manager.attr_num_epochs,
            status=0
        )

    def on_train_end(self, prediction_batch: np.ndarray, true_batch: np.ndarray):
        """Update iteration progress bar"""
        self._progress_bar.update(self._progress_bar_iterations,
                                  advance=self.iteration_manager.attr_tr_size,
                                  loss=self.loss.get_last_tr_loss(),
                                  status=self.iteration_manager.attr_it_tr
                                  )

    def on_epoch_end(self):
        """Update epoch progress bar

        Args:
            loss:float, loss of the model
            epoch, int
            **kwargs:

        Returns:
        """
        self._progress_bar.update(
            self._progress_bar_epochs,
            advance=1,
            loss=self.loss.get_last_tr_loss(),
            status=self.iteration_manager.attr_last_epoch
        )

    @property
    def progress_bar(self) -> Progress:
        return self._progress_bar
