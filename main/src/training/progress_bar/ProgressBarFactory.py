from typing import Optional

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.IterationManager import IterationManager
from main.src.training.metrics.losses.AbstractLoss import AbstractLoss
from main.src.training.progress_bar.ProgressBar0 import ProgressBar0
from main.src.training.progress_bar.ProgressBar1 import ProgressBar1


class ProgressBarFactory(BaseClass):
    """Choose the correct progress bar based on the length availability of the length information (None = not available)"""

    @staticmethod
    def create(length: Optional[int], iteration_manager: IterationManager,loss:AbstractLoss):
        """
        Args:
            length: Optional[int] length of the dataset if available else None
            num_epochs: int, number of epochs
        """
        if length is not None:
            return ProgressBar0(length, iteration_manager,loss)
        else:
            return ProgressBar1(iteration_manager,loss)
