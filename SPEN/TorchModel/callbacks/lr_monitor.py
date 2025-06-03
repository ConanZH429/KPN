import torch
import rich

from .callback import Callback


class LRMonitor(Callback):
    def __init__(self,
                 verbose: bool = True,
                 ):
        super().__init__()
        self.verbose = verbose
    
    def on_train_epoch_start(self, trainer):
        lr = trainer.optimizer.param_groups[0]["lr"]
        trainer.log_dict({"lr": lr}, epoch=trainer.now_epoch)
        if self.verbose:
            rich.print(f"Learning rate: {lr}")
        