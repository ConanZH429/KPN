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
            args=config.backbone_args[config.backbone]
        )
        # neck
        neck_factory = NeckFactory()
        self.neck = neck_factory.create_neck(
            config.neck,
            in_channels=self.backbone.out_channels,
            args=config.neck_args[config.neck],
        )
        # head
        head_factory = HeadFactory()
        self.head = head_factory.create_head(
            config.head,
            in_channels=self.neck.out_channels,
            args=config.head_args[config.head],
        )
        # pose head
        pose_head_factory = PoseHeadFactory()
        self.pos_head, self.ori_head = pose_head_factory.create_pose_head(
            pos_type=config.pos_type,
            pos_args=config.pos_args[config.pos_type],
            ori_type=config.ori_type,
            ori_args=config.ori_args[config.ori_type],
            pos_feature_dims=self.head.pos_feature_dims,
            ori_feature_dims=self.head.ori_feature_dims,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        pos_feature, ori_feature = self.head(x)
        pos_pre_dict = self.pos_head(pos_feature)
        ori_pre_dict = self.ori_head(ori_feature)
        return pos_pre_dict, ori_pre_dict