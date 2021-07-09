from typing import Optional

from main.src.param_savers.BaseClass import BaseClass
from main.src.training.progress_bar.ProgressBar0 import ProgressBar0
from main.src.training.progress_bar.ProgressBar1 import ProgressBar1


class ProgressBarFactory(BaseClass):
    def __init__(self, length: Optional[int], num_epochs: int):
        if length is not None:
            self.progress_bar = ProgressBar0(length, num_epochs)
        else:
            self.progress_bar = ProgressBar1(num_epochs)

    def end_epoch(self, loss: int, epoch: int, img_processed: int, **kwargs):
        self.progress_bar.end_epoch(loss=loss, epoch=epoch, img_processed=img_processed, **kwargs)

    def end_iteration(self, loss: float, tr_batch_size: int, it_tr: int, img_processed: int, **kwargs):
        self.progress_bar.end_iteration(loss=loss, tr_batch_size=tr_batch_size, it_tr=it_tr,
                                        img_processed=img_processed, **kwargs)

    def __enter__(self, *args, **kwargs):
        return self.progress_bar.__enter__()

    def __exit__(self, *args, **kwargs):
        return self.progress_bar.__exit__(*args, **kwargs)