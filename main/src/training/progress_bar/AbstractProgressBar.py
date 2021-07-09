from abc import ABC, abstractmethod

from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TaskID

from main.src.param_savers.BaseClass import BaseClass


class AbstractProgressBar( ABC):
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
        raise NotImplementedError

    @property
    def progress_bar_epochs(self) -> TaskID:
        raise NotImplementedError

    @property
    def progress_bar(self) -> Progress:
        raise NotImplementedError

    def __enter__(self, *args, **kwargs):
        return self.progress_bar.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        return self.progress_bar.__exit__(*args, **kwargs)

    @abstractmethod
    def end_epoch(self, loss: int, epoch: int, **kwargs):
        return NotImplementedError

    @abstractmethod
    def end_iteration(self, loss: float, status: int, **kargs):
        return NotImplementedError
