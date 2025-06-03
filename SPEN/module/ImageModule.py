import rich
import rich.table
import torch
import math
import pickle
import torch.nn as nn
from torch import Tensor
from pathlib import Path
from collections import OrderedDict

from ..TorchModel import Model

from ..model import SPEN
from ..cfg import SPEEDConfig
from ..utils import PosLossFactory, OriLossFactory
from ..utils import PosLossMetric, OriLossMetric, LossMetric, PosErrorMetric, OriErrorMetric, ScoreMetric
from ..pose import get_ori_decoder, get_pos_decoder
from ..pose import PosTransform, OriTransform

from typing import Dict, Union, List



class ImageModule(Model):
    def __init__(self, config: Union[SPEEDConfig]):
        super().__init__()
        # config
        self.config = config
        # model
        self.model = SPEN(self.config)
        # pose
        self.pos_decoder = get_pos_decoder(config.pos_type, **config.pos_args[config.pos_type])
        self.ori_decoder = get_ori_decoder(config.ori_type, **config.ori_args[config.ori_type])
        # transform
        self.pos_transform = PosTransform(config.pos_type, **config.pos_args[config.pos_type])
        self.ori_transform = OriTransform(config.ori_type, **config.ori_args[config.ori_type])
        if self.config.compile:
            self.pos_transform = torch.compile(self.pos_transform, mode="reduce-overhead", fullgraph=True)
            self.ori_transform = torch.compile(self.ori_transform, mode="reduce-overhead", fullgraph=True)

        self.test_result_dict = {}

        self._loss_init()

        self._metrics_init()


    def on_fit_start(self):
        # hyperparams
        self.trainer.logger.log_hyperparams(self.config)
        # code
        father_folder = Path("./SPEN")
        for file in father_folder.rglob("*.py"):
            self.trainer.logger.log_code(file_path=file)
        # datasetsplit
        if self.config.dataset == "SPEED":
            father_folder = Path(".").resolve().parent
            dataset_folder = father_folder / "datasets" / "speed"
            self.trainer.logger.log_file(str(dataset_folder / "train.txt"))
            self.trainer.logger.log_file(str(dataset_folder / "val.txt"))
            self.trainer.logger.log_file(str(dataset_folder / "train_label.json"))
            self.trainer.logger.log_file(str(dataset_folder / "val_label.json"))


    def forward(self, x):
        return self.model(x)


    def train_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        pos_pre_dict, ori_pre_dict = self.forward(images)
        # transform
        pos_pre_dict = self.pos_transform.transform(pos_pre_dict)
        ori_pre_dict = self.ori_transform.transform(ori_pre_dict)
        # loss
        pos_loss_list = [
            loss(pos_pre_dict, labels["pos_encode"], now_epoch=self.trainer.now_epoch) for loss in self.pos_loss_list
        ]
        ori_loss_list = [
            loss(ori_pre_dict, labels["ori_encode"], now_epoch=self.trainer.now_epoch) for loss in self.ori_loss_list
        ]
        pos_loss = torch.tensor(0.0, requires_grad=True).to(images.device)
        for loss_dict in pos_loss_list:
            for k in loss_dict.keys():
                pos_loss += loss_dict[k]
        ori_loss = torch.tensor(0.0, requires_grad=True).to(images.device)
        for loss_dict in ori_loss_list:
            for k in loss_dict.keys():
                ori_loss += loss_dict[k]
        train_loss = pos_loss + ori_loss
        # metrics
        self._update_train_metrics(num_samples, pos_loss_list, ori_loss_list, train_loss)
        self._train_log(log_online=False)
        return train_loss
    
    
    def on_train_epoch_end(self):
        self._train_log(log_online=True)
        self._train_metrics_reset()

    
    def val_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        pos_pre_dict, ori_pre_dict = self.forward(images)
        # transform
        pos_pre_dict = self.pos_transform.transform(pos_pre_dict)
        ori_pre_dict = self.ori_transform.transform(ori_pre_dict)
        self._update_val_metrics(num_samples,
                                 pos_pre_dict["cart"], labels["pos"],
                                 ori_pre_dict["quat"], labels["ori"])
        self._val_log(log_online=False)
    

    def on_val_epoch_end(self):
        self._val_log(log_online=True)
        self._val_metrics_reset()
    

    def on_test_start(self):
        self.test_result_dict = {}


    def test_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        pos_pre_dict, ori_pre_dict = self.forward(images)
        # transform
        pos_pre_dict = self.pos_transform.transform(pos_pre_dict)
        ori_pre_dict = self.ori_transform.transform(ori_pre_dict)
        self.test_result_dict[labels["image_name"][0]] = {
            "pos_label": labels["pos"][0].cpu().numpy(),
            "ori_label": labels["ori"][0].cpu().numpy(),
            "pos_encode_label": {
                k: v[0].cpu().numpy() for k, v in labels["pos_encode"].items()
            },
            "ori_encode_label": {
                k: v[0].cpu().numpy() for k, v in labels["ori_encode"].items()
            },
            "pos_encode_pred": {
                k: v[0].cpu().numpy() for k, v in pos_pre_dict.items()
            },
            "ori_encode_pred": {
                k: v[0].cpu().numpy() for k, v in ori_pre_dict.items()
            },
            "pos_pred": pos_pre_dict["cart"][0].cpu().numpy(),
            "ori_pred": ori_pre_dict["quat"][0].cpu().numpy(),
        }
        self._update_test_metrics(num_samples,
                                  pos_pre_dict["cart"], labels["pos"],
                                  ori_pre_dict["quat"], labels["ori"])
        self._test_log(log_online=True)
    

    def on_test_end(self):
        pickle_path = Path(self.trainer.callbacks[0].dirpath) / "result.pkl"
        with open(pickle_path, "wb") as f:
            pickle.dump(self.test_result_dict, f)
        self.trainer.logger.log_file(str(pickle_path))
        self._test_log(log_online=True)
        self._test_metrics_reset()


    def _loss_init(self):
        # loss
        ## pos_loss
        pos_loss_factory = PosLossFactory()
        self.pos_loss_list = nn.ModuleList()
        for pos_type, loss_type in self.config.pos_loss_dict.items():
            pos_loss =  pos_loss_factory.create_pos_loss(
                pos_type=pos_type,
                loss_type=loss_type,
                beta=self.config.pos_loss_args[pos_type]["beta"],
                weight_strategy=self.config.pos_loss_args[pos_type]["weight_strategy"],
            )
            if self.config.compile:
                pos_loss = torch.compile(pos_loss, mode="reduce-overhead", fullgraph=True)
            self.pos_loss_list.append(pos_loss)
        ## ori_loss
        ori_loss_factory = OriLossFactory()
        self.ori_loss_list = nn.ModuleList()
        for ori_type, loss_type in self.config.ori_loss_dict.items():
            ori_loss =  ori_loss_factory.create_ori_loss(
                ori_type=ori_type,
                loss_type=loss_type,
                beta=self.config.ori_loss_args[ori_type]["beta"],
                weight_strategy=self.config.ori_loss_args[ori_type]["weight_strategy"],
            )
            if self.config.compile:
                ori_loss = torch.compile(ori_loss, mode="reduce-overhead", fullgraph=True)
            self.ori_loss_list.append(ori_loss)

    
    def _metrics_init(self):
        self.train_pos_loss_metric_list = nn.ModuleList([
            PosLossMetric(pos_type=pos_type) for pos_type in self.config.pos_loss_dict.keys()
        ])
        self.train_ori_loss_metric_list = nn.ModuleList([
            OriLossMetric(ori_type=ori_type) for ori_type in self.config.ori_loss_dict.keys()
        ])
        self.train_loss_metric = LossMetric()

        self.pos_error_metric = PosErrorMetric()
        self.ori_error_metric = OriErrorMetric()
        self.score_metric = ScoreMetric(self.config.ALPHA)


    def _update_train_metrics(self,
                              num_samples: int,
                              pos_loss_list: List[Dict[str, Tensor]],
                              ori_loss_list: List[Dict[str, Tensor]],
                              loss: Tensor):
        for i in range(len(self.train_pos_loss_metric_list)):
            self.train_pos_loss_metric_list[i].update(pos_loss_list[i], num_samples)
        for i in range(len(self.train_ori_loss_metric_list)):
            self.train_ori_loss_metric_list[i].update(ori_loss_list[i], num_samples)
        self.train_loss_metric.update(loss, num_samples)


    def _train_log(self, log_online):
        data = {}
        data.update({"loss": self.train_loss_metric.compute()})
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="train",
                      log_online=log_online)
        data = {}
        # loss
        for loss in self.train_pos_loss_metric_list:
            data.update(loss.compute())
        for loss in self.train_ori_loss_metric_list:
            data.update(loss.compute())
        if not log_online:
            return
        # beta
        for loss in self.pos_loss_list:
            data.update(
                {beta_name: loss.beta_dict[beta_name].beta
                 for beta_name in loss.beta_dict.keys()}
            )
        for loss in self.ori_loss_list:
            data.update(
                {beta_name: loss.beta_dict[beta_name].beta
                 for beta_name in loss.beta_dict.keys()}
            )
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=False,
                      prefix="train",
                      log_online=log_online)
    

    def _train_metrics_reset(self):
        for loss in self.train_pos_loss_metric_list:
            loss.reset()
        for loss in self.train_ori_loss_metric_list:
            loss.reset()
        self.train_loss_metric.reset()


    def _update_val_metrics(self, num_samples: int,
                                  pos_decode: Tensor, pos_label: Tensor,
                                  ori_decode: Tensor, ori_label: Tensor):
        self.pos_error_metric.update(pos_decode, pos_label, num_samples)
        self.ori_error_metric.update(ori_decode, ori_label, num_samples)
        self.score_metric.update(self.pos_error_metric.compute()[1], self.ori_error_metric.compute())


    def _val_log(self, log_online):
        data = {}
        pos_error = self.pos_error_metric.compute()
        data.update({
            "pos_error": pos_error[0],
            "Et": pos_error[1],
            "ori_error": self.ori_error_metric.compute(),
            "score": self.score_metric.compute(),
        })
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="val",
                      log_online=log_online)
    

    def _val_metrics_reset(self):
        self.pos_error_metric.reset()
        self.ori_error_metric.reset()
        self.score_metric.reset()


    def _update_test_metrics(self, num_samples: int,
                                   pos_decode: Tensor, pos_label: Tensor,
                                   ori_decode: Tensor, ori_label: Tensor):
        self.pos_error.update(pos_decode, pos_label, num_samples)
        self.ori_error.update(ori_decode, ori_label, num_samples)
        self.score.update(self.pos_error.compute()[1], self.ori_error.compute())
    

    def _test_log(self, log_online):
        data = {}
        pos_error = self.pos_error_metric.compute()
        data.update({
            "pos_error": pos_error[0],
            "Et": pos_error[1],
            "ori_error": self.ori_error_metric.compute(),
            "score": self.score_metric.compute(),
        })
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="test",
                      log_online=log_online)
    
    
    def _test_metrics_reset(self):
        self.pos_error_metric.reset()
        self.ori_error_metric.reset()
        self.score_metric.reset()