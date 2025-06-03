from pathlib import Path
from typing import List, Tuple, Union

class Config:
    def __init__(self):
        # global config
        self.exp_type: str = ""
        self.seed: int = 0
        self.benchmark: bool = False
        self.debug: bool = False

        # logger
        self.comet_api: str = ""
        self.offline: bool = False

        # dataset
        self.dataset_folder: Union[Path, str] = ""
        self.cache: bool = False
        
        # train
        self.epochs: int
        self.batch_size: int
        self.num_workers: int
        self.optimizer: str
        self.scheduler: str
        self.lr0: float
        self.lr_min: float
        self.weight_decay: float
        self.momentum: float
        self.gradient_clip_val: Union[float, None] = None

        # model

        # loss

        # metrics

        # data augmentation