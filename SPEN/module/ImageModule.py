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

from .utils import KeypointsDistanceCalculator, PoseDecoder
from ..model import SPEN
from ..cfg import SPEEDplusConfig
from ..utils import get_keypoints_loss, get_uncertainty_loss
from ..utils import LossMetric, PosErrorMetric, OriErrorMetric, ScoreMetric, KeypointsErrorMetric
from ..keypoints import get_keypoints_decoder

from typing import Dict, Union, List



class ImageModule(Model):
    def __init__(self, config: Union[SPEEDplusConfig]):
        super().__init__()
        # config
        self.config = config
        # model
        self.model = SPEN(self.config)
        # keypoints
        self.keypoints_decoder = get_keypoints_decoder(
            config.keypoints_type,
            input_image_shape=config.image_size,
            **config.keypoints_args[config.keypoints_type],
        )
        self.keypoints_distance_calculator = KeypointsDistanceCalculator(
            image_size=config.image_size,
        )
        self.pose_decoder = PoseDecoder(
            image_size=config.image_size,
            points_world=config.keypoints,
            device=self.config.device,
        )

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


    def forward(self, x):
        return self.model(x)


    def train_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        uncertainty_pre, keypoints_encode_pre = self.forward(images)     # B, N  B, N, H, W
        # loss
        keypoints_encode_loss = self.keypoints_loss(keypoints_encode_pre, labels["keypoints_encode"], labels["points_vis"], now_epoch=self.trainer.now_epoch)
        keypoints_decode_pre = self.keypoints_decoder.decode(keypoints_encode_pre)  # B, N, 2
        uncertainty_label = self.keypoints_distance_calculator(keypoints_decode_pre, labels["points_image"])
        uncertainty_loss = self.uncertainty_loss(uncertainty_pre, uncertainty_label, now_epoch=self.trainer.now_epoch)
        train_loss = keypoints_encode_loss + uncertainty_loss
        # metrics
        self._update_train_metrics(num_samples, keypoints_encode_loss, uncertainty_loss, train_loss)
        self._train_log(log_online=False)
        return train_loss
    
    
    def on_train_epoch_end(self):
        self._train_log(log_online=True)
        self._train_metrics_reset()

    
    def val_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        uncertainty_pre, keypoints_encode_pre = self.forward(images)     # B, N  B, N, H, W
        # decode keypoints
        keypoints_decode_pre = self.keypoints_decoder.decode(keypoints_encode_pre)  # B, N, 2
        # decode to position and orientation
        pos_pre, ori_pre = self.pose_decoder.decode2pose(
            keypoints_decode_pre=keypoints_decode_pre,
            uncertainty_pre=uncertainty_pre
        )
        self._update_val_metrics(num_samples,
                                 keypoints_decode_pre, labels["points_image"],
                                 labels["points_vis"],
                                 pos_pre, labels["pos"],
                                 ori_pre, labels["ori"])
        self._val_log(log_online=False)
    

    def on_val_epoch_end(self):
        self._val_log(log_online=True)
        self._val_metrics_reset()
    

    def on_test_start(self):
        self.test_result_dict = {}


    def test_step(self, index, batch):
        images, _, labels = batch
        num_samples = images.size(0)
        uncertainty_pre, keypoints_encode_pre = self.forward(images)     # B, N  B, N, H, W
        # decode keypoints
        keypoints_decode_pre = self.keypoints_decoder.decode(keypoints_encode_pre)  # B, N, 2
        uncertainty_label = self.keypoints_distance_calculator(keypoints_decode_pre, labels["points_image"])
        # decode to position and orientation
        pos_pre, ori_pre = self.pose_decoder.decode2pose(
            keypoints_decode_pre=keypoints_decode_pre,
            uncertainty_pre=uncertainty_pre
        )
        self.test_result_dict[labels["image_name"][0]] = {
            "pos_label": labels["pos"][0].cpu().numpy(),
            "ori_label": labels["ori"][0].cpu().numpy(),
            "keypoints_encode_label": labels["keypoints_encode"][0].cpu().numpy(),
            "keypoints_encode_pred": keypoints_encode_pre[0].cpu().numpy(),
            "keypoints_decode_label": labels["points_image"][0].cpu().numpy(),
            "keypoints_decode_pred": keypoints_decode_pre[0].cpu().numpy(),
            "uncertainty_label": uncertainty_label[0].cpu().numpy(),
            "uncertainty_pred": uncertainty_pre[0].cpu().numpy(),
            "pos_pred": pos_pre[0].cpu().numpy(),
            "ori_pred": ori_pre[0].cpu().numpy(),
        }
        self._update_test_metrics(num_samples,
                                  keypoints_decode_pre, labels["points_image"],
                                  labels["points_vis"],
                                  pos_pre, labels["pos"],
                                  ori_pre, labels["ori"])
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
        ## keypoints loss
        self.keypoints_loss = get_keypoints_loss(
            self.config.keypoints_type,
            self.config.keypoints_loss_type,
            self.config.keypoints_beta,
            self.config.keypoints_weight_strategy,
            **self.config.keypoints_loss_args[self.config.keypoints_loss_type]
        )
        ## uncertainty loss
        self.uncertainty_loss = get_uncertainty_loss(
            self.config.uncertainty_type,
            self.config.uncertainty_loss_type,
            self.config.uncertainty_beta,
            self.config.uncertainty_weight_strategy,
            **self.config.uncertainty_loss_args[self.config.uncertainty_loss_type]
        )


    def _metrics_init(self):
        self.keypoints_loss_metric = LossMetric()
        self.uncertainty_loss_metric = LossMetric()
        self.train_loss_metric = LossMetric()

        self.keypoints_error_metric = KeypointsErrorMetric()
        self.pos_error_metric = PosErrorMetric()
        self.ori_error_metric = OriErrorMetric()
        self.score_metric = ScoreMetric(self.config.ALPHA)


    def _update_train_metrics(self,
                              num_samples: int,
                              keypoints_encode_loss: Tensor,
                              uncertainty_loss: Tensor,
                              loss: Tensor):
        self.keypoints_loss_metric.update(keypoints_encode_loss, num_samples)
        self.uncertainty_loss_metric.update(uncertainty_loss, num_samples)
        self.train_loss_metric.update(loss, num_samples)


    def _train_log(self, log_online):
        data = {}
        data.update(
            {
                "loss": self.train_loss_metric.compute(),
                "keypoints_loss": self.keypoints_loss_metric.compute(),
                "uncertainty_loss": self.uncertainty_loss_metric.compute(),
            }
        )
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=True,
                      prefix="train",
                      log_online=log_online)
        if not log_online:
            return
        data = {
            "keypoints_loss_beta": self.keypoints_loss.beta.beta,
            "uncertainty_loss_beta": self.uncertainty_loss.beta.beta,
        }
        self.log_dict(data=data,
                      epoch=self.trainer.now_epoch,
                      on_bar=False,
                      prefix="train",
                      log_online=log_online)
    

    def _train_metrics_reset(self):
        self.keypoints_loss_metric.reset()
        self.uncertainty_loss_metric.reset()
        self.train_loss_metric.reset()


    def _update_val_metrics(self, num_samples: int,
                                  keypoints_decode_pre: Tensor, keypoints_decode_label: Tensor,
                                  points_vis: Tensor,
                                  pos_pre: Tensor, pos_label: Tensor,
                                  ori_pre: Tensor, ori_label: Tensor):
        self.keypoints_error_metric.update(keypoints_decode_pre, keypoints_decode_label, points_vis, num_samples)
        self.pos_error_metric.update(pos_pre, pos_label, num_samples)
        self.ori_error_metric.update(ori_pre, ori_label, num_samples)
        self.score_metric.update(self.pos_error_metric.compute()[1], self.ori_error_metric.compute())


    def _val_log(self, log_online):
        data = {}
        pos_error = self.pos_error_metric.compute()
        data.update({
            "keypoints_error": self.keypoints_error_metric.compute(),
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
        self.keypoints_error_metric.reset()
        self.pos_error_metric.reset()
        self.ori_error_metric.reset()
        self.score_metric.reset()


    def _update_test_metrics(self, num_samples: int,
                                  keypoints_decode_pre: Tensor, keypoints_decode_label: Tensor,
                                  points_vis: Tensor,
                                  pos_pre: Tensor, pos_label: Tensor,
                                  ori_pre: Tensor, ori_label: Tensor):
        self.keypoints_error_metric.update(keypoints_decode_pre, keypoints_decode_label, points_vis, num_samples)
        self.pos_error_metric.update(pos_pre, pos_label, num_samples)
        self.ori_error_metric.update(ori_pre, ori_label, num_samples)
        self.score_metric.update(self.pos_error_metric.compute()[1], self.ori_error_metric.compute())
    

    def _test_log(self, log_online):
        data = {}
        pos_error = self.pos_error_metric.compute()
        data.update({
            "keypoints_error": self.keypoints_error_metric.compute(),
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
        self.keypoints_error_metric.reset()
        self.pos_error_metric.reset()
        self.ori_error_metric.reset()
        self.score_metric.reset()