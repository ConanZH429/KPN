import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.layers.conv_bn_act import ConvBnAct
from pathlib import Path

from .backbone import *
from .neck import *
from .head import *
from .pose_head import *

from ..cfg import SPEEDConfig, SPEEDplusConfig

from typing import Union


class SPEN(nn.Module):

    def __init__(self, config: Union[SPEEDConfig, SPEEDplusConfig]):
        super().__init__()
        # backbone
        backbone_factory = BackboneFactory()
        self.backbone = backbone_factory.create_backbone(
            config.backbone,
            pretrained=True,
            **config.backbone_args[config.backbone]
        )
        # neck
        neck_factory = NeckFactory()
        self.neck = neck_factory.create_neck(
            config.neck,
            in_channels=self.backbone.out_channels,
            **config.neck_args[config.neck],
        )
        # head
        head_factory = HeadFactory()
        self.head = head_factory.create_head(
            config.head,
            in_channels=self.neck.out_channels,
            **config.head_args[config.head],
        )
        # keypoints head
        keypoints_head_factory = KeypointsHeadFactory()
        self.keypoints_head = keypoints_head_factory.create_keypoints_head(
            config.keypoints_type,
            config.keypoints_loss_type,
            self.head.keypoints_feature_dims,
            config.keypoints.shape[0],
            **config.keypoints_args[config.keypoints_type]
        )
        # uncertainty head
        uncertainty_head_factory = UncertaintyHeadFactory()
        self.uncertainty_head = uncertainty_head_factory.create_uncertainty_head(
            config.uncertainty_type,
            config.uncertainty_loss_type,
            self.head.uncertainty_feature_dims,
            config.keypoints.shape[0],
            **config.uncertainty_args[config.uncertainty_type]
        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        uncertainty_feature, keypoints_feature = self.head(x)
        uncertainty = self.uncertainty_head(uncertainty_feature)
        keypoints_encode = self.keypoints_head(keypoints_feature)
        return uncertainty, keypoints_encode