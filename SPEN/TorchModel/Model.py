import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def train_step(self, index, batch):
        raise NotImplementedError("train_step method is not implemented")

    def val_step(self, index, batch):
        raise NotImplementedError("val_step method is not implemented")
    
    def on_fit_start(self):
        """Called at the beginning of the fit method"""
    
    def on_train_epoch_start(self):
        """Called at the beginning of each train epoch"""

    def on_train_epoch_end(self):
        """Called at the end of each train epoch"""
    
    def on_val_epoch_end(self):
        """Called at the end of each validation epoch"""
    
    def on_fit_epoch_end(self):
        """Called at the end of each fit epoch"""

    def on_test_start(self):
        """Called at the beginning of the test method"""
    
    def log_dict(self,
                 data: dict,
                 epoch: int,
                 on_bar: bool = False,
                 prefix: str = "",
                 prefix_on_bar: str = "",
                 log_online: bool = True,
                 ):
        self.trainer.log_dict(data=data,
                              epoch=epoch,
                              on_bar=on_bar,
                              prefix=prefix,
                              prefix_on_bar=prefix_on_bar,
                              log_online=log_online
                              )