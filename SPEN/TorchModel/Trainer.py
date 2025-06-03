import torch
from torch import Tensor
import rich

from datetime import datetime
from tqdm import tqdm
from typing import Union, Optional, Dict, List

from .utils import get_optimizer
from .utils import get_scheduler
from .utils import Config

from .logger import Logger
from .callbacks import Callback


class Trainer:
    def __init__(self,
                 model,
                 config,
                 logger: Logger,
                 callbacks: List = [],
                 device: str = "cpu",
                 gradient_accumulation_steps: int = 1,
                 gradient_clip_val: Optional[float] = None,
                 ):
        """
        Initialize the Trainer class

        Args:
            model (torch.nn.Module): The model to train
            config (Config): The configuration object
            logger (Logger): The logger to use
            callbacks (Dict[str, List]): The callbacks to use
            precision (str, optional): The precision to use. Defaults to "no".
            device (str, optional): The device to use. Defaults to "cpu".
            gradient_accumulation_steps (int, optional): The number of gradient accumulation steps. Defaults to 1.
            gradient_clip_val (Optional[float], optional): The gradient clip value. Defaults to None.
        """
        self.model = model
        self.model.to(device)
        self.config = config
        self.device = device
        optimizer = config.optimizer
        lr_scheduler = config.scheduler
        self.optimizer = get_optimizer(self.model, optimizer, config)
        self.lr_scheduler = get_scheduler(lr_scheduler, self.optimizer, config)
        self.callbacks = callbacks
        self.gradient_clip_val = gradient_clip_val
        self.callbacks: List[Callback] = callbacks
        self.scaler = torch.amp.GradScaler(device=self.device)
        self.autocast = torch.amp.autocast(device_type=self.device, dtype=torch.float16)
        self.max_epochs = config.epochs
        self.now_epoch = None
        self.logger = logger
        self.model.trainer = self
        self.postfix_dict = {}
        self.metrics_dict = {}
        if self.config.benchmark:
            torch.backends.cudnn.benchmark = True
    

    def fit(self, train_loader, valid_loader):
        """
        Fit the model to the given data

        Args:
            train_loader: The training data loader
            valid_loader: The validation data loader
        """
        device_type = '🐌' if 'cpu' in self.device else ('⚡️' if 'cuda' in self.device else '🚀')
        rich.print(f"<<<<<<<< Training on {self.device} {device_type} >>>>>>>>")

        self.train_loader, self.valid_loader = train_loader, valid_loader

        self.on_fit_start()
        
        self.fit_loop()
    

    def fit_loop(self):
        for epoch in range(1, self.max_epochs+1):
            self.now_epoch = epoch
            nowtime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info = f"{nowtime} - Epoch {epoch}/{self.max_epochs}"
            rich.print(f"{info:=^50}")

            self.train_epoch_loop()

            self.val_epoch_loop()

            if self.lr_scheduler is not None:
                if self.lr_scheduler.__class__.__name__ in {"ReduceLROnPlateau", "ReduceWarmupCosinLR"}:
                    self.lr_scheduler.step(self.metrics_dict["val/score"])
                else:
                    self.lr_scheduler.step()
            
            self.on_fit_epoch_end()
    

    def train_epoch_loop(self):
        self.on_train_epoch_start()

        batch_loop = tqdm(self.train_loader,
                          desc="Train",)
        self.postfix_dict = {}

        self.model.train()

        index = 0
        for batch in batch_loop:
            batch = self.to_device(batch)
            index += 1
            self.optimizer.zero_grad()

            with self.autocast:
                loss = self.model.train_step(index, batch)

            self.scaler.scale(loss).backward()

            if self.gradient_clip_val is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            batch_loop.set_postfix(self.postfix_dict)
        
        self.on_train_epoch_end()
    
    
    def val_epoch_loop(self):
        batch_loop = tqdm(self.valid_loader,
                          desc="Val",)
        self.postfix_dict = {}
        
        self.model.eval()
        index = 0
        for batch in batch_loop:
            batch = self.to_device(batch)
            index += 1
            with self.autocast:
                with torch.no_grad():
                    self.model.val_step(index, batch)
        
            batch_loop.set_postfix(self.postfix_dict)
        
        self.on_val_epoch_end()
    

    def test(self, test_loader, weight_path = None):
        """
        Test the model on the given data
        
        Args:
            test_loader: The test data loader
        """
        device_type = '🐌' if 'cpu' in self.device else ('⚡️' if 'cuda' in self.device else '🚀')
        rich.print(f"<<<<<<<< Testing on {self.device} {device_type} >>>>>>>>")

        self.weight_path = weight_path
        self.test_loader = test_loader

        # load
        if self.weight_path:
            pth = torch.load(self.weight_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(pth, strict=True)

        self.on_test_start()

        self.test_loop()

        self.on_test_end()
    
    
    def test_loop(self):
        loop = tqdm(self.test_loader,
                    desc="Test",)
        self.postfix_dict = {}

        self.model.eval()
        index = 0
        for batch in loop:
            batch = self.to_device(batch)
            index += 1
            with self.autocast:
                with torch.no_grad():
                    self.model.test_step(index, batch)
    
            loop.set_postfix(self.postfix_dict)


    def on_fit_start(self):
        [cb.on_fit_start(trainer=self) for cb in self.callbacks]
        self.model.on_fit_start()
    
    
    def on_train_epoch_start(self):
        [cb.on_train_epoch_start(trainer=self) for cb in self.callbacks]
        self.model.on_train_epoch_start()


    def on_train_epoch_end(self):
        [cb.on_train_epoch_end(trainer=self) for cb in self.callbacks]
        self.model.on_train_epoch_end()


    def on_val_epoch_end(self):
        [cb.on_val_epoch_end(trainer=self) for cb in self.callbacks]
        self.model.on_val_epoch_end()
    
    def on_fit_epoch_end(self):
        [cb.on_fit_epoch_end(trainer=self) for cb in self.callbacks]
        self.model.on_fit_epoch_end()
    
    def on_test_start(self):
        [cb.on_test_start(trainer=self) for cb in self.callbacks]
        self.model.on_test_start()

    def on_test_end(self):
        [cb.on_test_end(trainer=self) for cb in self.callbacks]
        self.model.on_test_end()
    
    
    def to_device(self, data):
        if isinstance(data, Tensor):
            return data.to(self.device)
        elif isinstance(data, list):
            return [self.to_device(d) for d in data]
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        else:
            return data
    
    
    def log_dict(self,
                 data: dict,
                 epoch: int,
                 on_bar: bool = False,
                 prefix: str = "",
                 prefix_on_bar: str = "",
                 log_online: bool = True
                 ):
        data = {k: v.item() if isinstance(v, Tensor) else v for k, v in data.items()}
        if prefix:
            data_log = {f"{prefix}/{key}": value for key, value in data.items()}
        else:
            data_log = data
        self.metrics_dict.update(data_log)
        if on_bar:
            if prefix_on_bar:
                data_on_bar = {f"{prefix_on_bar}/{key}": value for key, value in data.items()}
            else:
                data_on_bar = data
            self.postfix_dict.update(data_on_bar)
        if log_online:
            self.logger.log_dict(data_log, epoch=epoch)