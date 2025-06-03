import torch
import rich

from pathlib import Path
from .callback import Callback


class Checkpoint(Callback):
    def __init__(self, 
                 dirpath: str,
                 filename: str,
                 monitor: str,
                 monitor_mode: str = "min",
                 save_last: bool = True,
                 verbose: bool = True
                 ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = Path(filename + ".pth")
        self.monitor = monitor
        self.monitor_fn = (lambda new, old: old is None or new < old) if monitor_mode == "min" else \
                            (lambda new, old: old is None or new > old)
        self.monitor_mode = monitor_mode
        self.save_last = save_last
        self.best = None
        self.best_epoch = None
        self.verbose = verbose
        self.best_str = ""
    
    def on_fit_epoch_end(self, trainer):
        metric = trainer.metrics_dict[self.monitor]
        # Save best model
        if self.monitor_fn(metric, self.best):
            self.best_str = "\n".join([f"{k}: {v}" for k, v in trainer.metrics_dict.items()])
            self.best = metric
            self.best_epoch = trainer.now_epoch
            torch.save(trainer.model.state_dict(), self.dirpath / self.filename)
            if self.verbose:
                rich.print(f"Save best model with {self.monitor}: {self.best} at {self.dirpath / self.filename}")
            best_dict = {
                "ori_error": trainer.metrics_dict["val/ori_error"],
                "pos_error": trainer.metrics_dict["val/pos_error"],
                "Et": trainer.metrics_dict["val/Et"],
            }
            trainer.log_dict(best_dict, trainer.now_epoch, prefix="best")
        # Save last model
        if self.save_last and trainer.now_epoch == trainer.config.epochs:
            last_path = self.dirpath / "last.pth"
            torch.save(trainer.model.state_dict(), last_path)
            if self.verbose:
                rich.print(f"Save last model at {last_path}")
            trainer.logger.log_file(str(last_path))
            best_path = self.dirpath / self.filename
            new_name = best_path.with_name(f"{self.filename.stem}-{self.best_epoch}.{self.filename.suffix}")
            best_path = best_path.rename(new_name)
            self.best_path = best_path
            trainer.logger.log_file(str(best_path))      # log best
            rich.print(f"Best model is at {new_name}")
            rich.print(self.best_str)
            trainer.logger.log_text(self.best_str)