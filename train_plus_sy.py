import comet_ml
import torch
import os

import torch.autograd.gradcheck
from SPEN.cfg import SPEEDplussyConfig
from SPEN.data import get_speedplussy_dataloader
from SPEN.module import ImageModule
from SPEN.utils import parse2config

from SPEN.TorchModel import Trainer
from SPEN.TorchModel.logger import CometLogger
from SPEN.TorchModel.callbacks import Checkpoint, LRMonitor, ModelSummary, Compile

from lightning.pytorch import seed_everything
from pathlib import Path


if __name__ == "__main__":
    config = SPEEDplussyConfig()
    config = parse2config(config)
    os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

    torch.set_float32_matmul_precision("high")
    
    dirpath = Path(f"./result/{config.name}")
    if not dirpath.exists():
        dirpath.mkdir(parents=True)
    
    seed_everything(config.seed)

    # ==========Trainer==========
    # ----------debug----------
    if config.debug:
        config.num_workers = 6
        config.cache = False
        config.batch_size = 10
        config.epochs = 10
        config.offline = True
    # ----------Callbacks----------
    checkpoint = Checkpoint(
        dirpath=str(dirpath),
        filename="best",
        monitor="val/score",
        monitor_mode="min"
    )
    lr_monitor = LRMonitor()
    model_summary = ModelSummary(input_size=(1, 1, *config.image_size), depth=4)
    # ----------Compile----------
    if config.compile:
        compile = Compile(mode="reduce-overhead", fullgraph=True)
        callbacks = [checkpoint, model_summary, lr_monitor, compile]
    else:
        callbacks = [checkpoint, model_summary, lr_monitor]
    # ----------Logger----------
    comet_logger = CometLogger(
        api_key=config.comet_api,
        project_name="paper3",
        experiment_name=config.name,
        online=not config.offline,
    )
    
    # ==========Model==========
    model = ImageModule(config=config)

    train_dataloader, val_dataloader, test_dataloader = get_speedplussy_dataloader(config)

    trainer = Trainer(
        model=model,
        config=config,
        callbacks=callbacks,
        device=config.device,
        logger=comet_logger,
        gradient_clip_val=config.gradient_clip_val,
    )

    trainer.fit(train_dataloader, val_dataloader)

    # test
    trainer.test(test_dataloader, weight_path=trainer.callbacks[0].best_path)