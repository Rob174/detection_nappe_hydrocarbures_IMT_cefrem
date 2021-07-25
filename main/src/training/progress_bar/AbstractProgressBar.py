from abc import ABC, abstractmethod

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskID

from main.src.param_savers.BaseClass import BaseClass


class AbstractProgressBar( ABC):
    """Base class to build a progressbar"""
    def __init__(self):
        self.columns = [
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
        ]

    @property
    def progress_bar_iterations(self) -> TaskID:
        """Returns the progress bar for iterations inside an epochs"""
        raise NotImplementedError

    @property
    def progress_bar_epochs(self) -> TaskID:
        """Returns the progress bar for epochs"""
        raise NotImplementedError

    @property
    def progress_bar(self) -> Progress:
        """Global progress bar object (with multiple progress bar potentially inside)"""
        raise NotImplementedError

    def __enter__(self, *args, **kwargs):
        """Context manager (with statement)"""
        return self.progress_bar.__enter__()

    def __exit__(self, *args, **kwargs):
        """Context manager (with statement)"""
        return self.progress_bar.__exit__(*args, **kwargs)

    @abstractmethod
    def end_epoch(self, loss: int, epoch: int, **kwargs):
        """Update method for the epoch progress bar"""
        return NotImplementedError

    @abstractmethod
    def end_iteration(self, loss: float, status: int, **kargs):
        """Update method for the iteration progressbar"""
        return NotImplementedError
