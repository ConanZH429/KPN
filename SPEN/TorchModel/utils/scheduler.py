import torch
import math
from torch.optim.lr_scheduler import LRScheduler

from .config import Config


class ReduceWarmupCosinLR(LRScheduler):
    def __init__(self, optimizer, warmup_epoch, max_epoch, lr0, lr_min, factor, threshold, patience):
        self.optimizer = optimizer
        self.last_epoch = 0
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        self.lr0 = lr0
        self.lr_min = lr_min
        self.factor = factor
        self.threshold = threshold
        self.patience = patience
        self.lambda_max = lr0 / lr0
        self.lambda_min = lr_min / lr0
        self.last_points = self.warmup_epoch
        self.best = 1e9
        self.num_bad_epochs = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self._initial_step()
    
    def step(self, metrics):
        current = float(metrics)
        if self.best - current > self.threshold:
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.num_bad_epochs > self.patience:
            self.lambda_max = self._last_lr[0] * self.factor / self.lr0
            self.last_points = self.last_epoch + 1
            self.num_bad_epochs = 0
        
        values = self.get_lr()
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
        
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def get_lr(self):
        now_epoch = self.last_epoch + 1
        self.last_epoch = now_epoch
        if now_epoch < self.warmup_epoch:
            return [self.warmup(now_epoch) * base_lr for base_lr in self.base_lrs]
        else:
            return [self.cos(now_epoch) * base_lr for base_lr in self.base_lrs]
    
    def _initial_step(self):
        self.optimizer._step_count = 0
        self.optimizer.param_groups[0]['lr'] = self.warmup(0) * self.base_lrs[0]

    def warmup(self, cur_iter):
        return self.lambda_min + (self.lambda_max - self.lambda_min) * cur_iter / (self.warmup_epoch - 1)

    def cos(self, cur_iter):
        return self.lambda_min + (self.lambda_max-self.lambda_min)*(1 + math.cos(math.pi * (cur_iter - self.last_points) / (self.max_epoch - self.last_points - 1))) / 2



def get_scheduler(sheduler_type: str, optimizer: torch.optim.Optimizer, config: Config) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get the learning rate scheduler

    Args:
        sheduler_type (str): The type of the scheduler to get
        optimizer (torch.optim.Optimizer): The optimizer
        config (Config): The config
    
    Returns:
        torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler
    """
    if hasattr(config, "lr_min"):
        lr_min = config.lr_min
    elif hasattr(config, "lr0"):
        lr_min = config.lr0 / 10
    else:
        lr_min = 0.0001
    warmup_epochs = config.warmup_epochs if hasattr(config, 'warmup_epochs') else 0
    epochs = config.epochs
    

    if sheduler_type == "WarmupCosin":
        lambda_max = config.lr0 / config.lr0
        lambda_min = config.lr_min / config.lr0
        warmup_epoch = config.warmup_epochs
        max_epoch = config.epochs
        lambda0 = lambda cur_iter: lambda_min + (lambda_max-lambda_min) * cur_iter / (warmup_epoch-1) if cur_iter < warmup_epoch \
            else lambda_min + (lambda_max-lambda_min)*(1 + math.cos(math.pi * (cur_iter - warmup_epoch) / (max_epoch - warmup_epoch - 1))) / 2
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    elif sheduler_type == "OnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode="min",
                                                               factor=1/2,
                                                               patience=10,
                                                               min_lr=lr_min,
                                                               threshold=0.01,
                                                               threshold_mode="abs")
    elif sheduler_type == "ReduceWarmupCosin":
        warmup_epoch = config.warmup_epochs
        max_epoch = config.epochs
        scheduler = ReduceWarmupCosinLR(optimizer,
                                        warmup_epoch=warmup_epoch,
                                        max_epoch=max_epoch,
                                        lr0=config.lr0,
                                        lr_min=lr_min,
                                        factor=3/4,
                                        threshold=0.01,
                                        patience=5)
    elif sheduler_type == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[15, 25],
            gamma=0.1,
            last_epoch=-1,
        )
    else:
        raise ValueError(f"Invalid scheduler type: {sheduler_type}")
    
    return scheduler