import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from timm.layers.mlp import Mlp, GluMlp
from timm.layers.conv_bn_act import ConvNormAct
from timm.models._efficientnet_blocks import UniversalInvertedResidual, InvertedResidual
from timm.models.vision_transformer import Block, Attention

from timm.layers.squeeze_excite import SEModule
from timm.layers.cbam import SpatialAttn, CbamModule

from timm.layers.weight_init import trunc_normal_

from typing import Optional, List, Tuple, Union, Literal


ConvAct = nn.Mish
MLPAct = nn.Mish


class DensFuse(nn.Module):
    def __init__(self, align_channels: int):
        super().__init__()
        self.conv_downsample = ConvNormAct(align_channels, align_channels, 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, size=current_feature[-2:], mode='bilinear', align_corners=True)
        return feature_fused


class SSIAFuse(nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        # spatial
        self.conv3x3 = ConvNormAct(in_channels[0], 1, 3, stride=2, dilation=1, act_layer=ConvAct)
        self.conv5x5 = ConvNormAct(in_channels[0], 1, 3, stride=2, dilation=2, act_layer=ConvAct)
        self.spatial_weight_conv = ConvNormAct(2, 1, 1, stride=1, act_layer=nn.Identity)

        # channel
        self.channel_weight_conv = nn.Sequential(
            nn.Conv2d(in_channels[-1]*2, in_channels[-1], 1),
            ConvAct(inplace=True),
            nn.Conv2d(in_channels[-1], in_channels[0] , 1),
        )

        # fuse
        self.conv_downsample = ConvNormAct(in_channels[0], in_channels[0], 3, stride=2, act_layer=ConvAct)
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        # spatial
        spatial_feature = torch.cat([
            self.conv3x3(shallow_feature),
            self.conv5x5(shallow_feature),
        ], dim=1)
        spatial_weight = F.sigmoid(self.spatial_weight_conv(spatial_feature))  # B, 1, H, W

        # channel
        avg_feature = F.adaptive_avg_pool2d(deep_feature, 1)
        max_feature = F.adaptive_max_pool2d(deep_feature, 1)
        channel_feature = torch.cat([avg_feature, max_feature], dim=1)  # B, C, 1, 1
        channel_weight = F.sigmoid(self.channel_weight_conv(channel_feature))  # B, C, 1, 1

        # fusion
        # feature_fused = self.conv_downsample(shallow_feature) + current_feature + F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
        feature_fused = torch.cat([
            self.conv_downsample(shallow_feature) * channel_weight,
            current_feature,
            F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True) * spatial_weight
        ], dim=1)
        return feature_fused


class Fuse(nn.Module):
    def __init__(self, in_channels: List[int]):
        super().__init__()
        self.conv_downsample = ConvNormAct(in_channels[0], in_channels[0], 3, stride=2, act_layer=ConvAct)
        self.attention = nn.Identity()
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        feature_fused = torch.cat([
            self.conv_downsample(shallow_feature),
            current_feature,
            F.interpolate(deep_feature, size=current_feature.shape[-2:], mode='bilinear', align_corners=True)
        ], dim=1)
        feature_fused = self.attention(feature_fused)
        return feature_fused


class SEFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = SEModule(sum(in_channels), rd_ratio=1./16, act_layer=ConvAct)


class SAMFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = SpatialAttn()

class CBAMFuse(Fuse):
    def __init__(self, in_channels: List[int]):
        super().__init__(in_channels)
        self.attention = CbamModule(sum(in_channels), rd_ratio=1./8, act_layer=ConvAct)


class AttFuse(nn.Module):
    def __init__(self, in_channels: List[int], att_type: Optional[str] = None):
        super().__init__()
        if att_type == "SE":
            self.attention = SEFuse(in_channels)
        elif att_type == "SAM":
            self.attention = SAMFuse(in_channels)
        elif att_type == "CBAM":
            self.attention = CBAMFuse(in_channels)
        elif att_type == "SSIA":
            self.attention = SSIAFuse(in_channels)
        elif att_type is None:
            self.attention = Fuse(in_channels)
        else:
            raise ValueError(f"Unknown attention type: {att_type}")
    
    def forward(self, shallow_feature: Tensor, current_feature: Tensor, deep_feature: Tensor):
        return self.attention(shallow_feature, current_feature, deep_feature)


class AvgPool(nn.Module):
    def __init__(self, pool_size: List[Tuple[int, int]]):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x: Tensor):
        return torch.cat([F.adaptive_avg_pool2d(x, ps).flatten(1) for ps in self.pool_size], dim=1)


class MaxPool(nn.Module):
    def __init__(self, pool_size: List[Tuple[int, int]]):
        super().__init__()
        self.pool_size = pool_size
    
    def forward(self, x: Tensor):
        return torch.cat([F.adaptive_max_pool2d(x, ps).flatten(1) for ps in self.pool_size], dim=1)


class MixPool(nn.Module):
    def __init__(self, pool_size: List[Tuple[int, int]]):
        super().__init__()
        self.pool_size = pool_size
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, x: Tensor):
        x_max_pooled = torch.cat([F.adaptive_max_pool2d(x, ps).flatten(1) for ps in self.pool_size], dim=1)
        x_avg_pooled = torch.cat([F.adaptive_avg_pool2d(x, ps).flatten(1) for ps in self.pool_size], dim=1)
        weight = F.softmax(self.weight, dim=0)
        return x_max_pooled * weight[0] + x_avg_pooled * weight[1]


class PatchEmbedding(nn.Module):
    def __init__(
            self,
            patch_shape: Union[Tuple[int, int], int, None] = None,
            embedding_mode: str = "mean"
    ):
        super().__init__()
        if isinstance(patch_shape, int):
            self.patch_shape = (patch_shape, patch_shape)
        elif isinstance(patch_shape, tuple):
            self.patch_shape = patch_shape
        elif patch_shape is None:
            self.patch_shape = None

        self.embedding_mode = embedding_mode

    def forward(self, x: Tensor):
        if self.patch_shape is not None:
            if self.embedding_mode == "mean":
                x = F.adaptive_avg_pool2d(x, self.patch_shape)  # B, C, H, W -> B, C, 1, 1
        x = x.flatten(2).transpose(1, 2)  # B, C, H, W -> B, C, H*W -> B, H*W, C
        return x


class WMSA(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.position_embedding = nn.Embedding(1500, dim)
        self.block = Block(dim=dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.weight_init()
    
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2) # B, C, H, W -> B, C, S -> B, S, C
        # position_token = torch.arange(0, x.size(1), device=x.device)   # S
        # position_embedded = self.position_embedding(position_token)        # S, C
        # x = x + position_embedded.unsqueeze(dim=0)                         # B, S, C
        # x = x.transpose(1, 2).reshape(B, C, H, W)       # B, S, C -> B, C, S -> B, C, H, W
        x = x.view(B, C, 2, H//2, 2, W//2)  # B, C, 2, H/2, 2, W/2
        x = x.permute(0, 2, 4, 1, 3, 5)  # B, 2, 2, C, H/2, W/2
        x = x.reshape(B*4, C, H//2, W//2)  # B*4, C, H/2, W/2
        x = x.flatten(2).transpose(1, 2)   # B*4, C, H/2, W/2 -> B*4, C, S -> B*4, S, C
        x = self.block(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2).reshape(B, 2, 2, C, H//2, W//2) # B*4, S, C -> B*4, C, S -> B, 2, 2, C, H/2, W/2
        x = x.permute(0, 3, 1, 4, 2, 5) # B, C, 2, H/2, 2, W/2
        x = x.reshape(B, C, H, W) # B, C, 2, H/2, 2, W/2 -> B, C, H, W
        return x

    def weight_init(self):
        # trunc_normal_(self.position_embedding.weight, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "init_weights"):
                m.init_weights()

class GMMSA(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # self.position_embedding = nn.Embedding(2000, dim)
        self.block = Block(dim=dim, num_heads=num_heads)
        self.layer_norm = nn.LayerNorm(dim)
        self.weight_init()
    
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        # x = x.flatten(2).transpose(1, 2) # B, C, H, W -> B, C, S -> B, S, C
        # position_token = torch.arange(0, x.size(1), device=x.device)   # S
        # position_embedded = self.position_embedding(position_token)        # S, C
        # x = x + position_embedded.unsqueeze(dim=0)                         # B, S, C
        # x = x.transpose(1, 2).reshape(B, C, H, W)       # B, S, C -> B, C, S -> B, C, H, W
        x = torch.cat([
            x[:, :, 0:H:2, 0:W:2],
            x[:, :, 0:H:2, 1:W:2],
            x[:, :, 1:H:2, 0:W:2],
            x[:, :, 1:H:2, 1:W:2],
        ], dim=0)   # B*4, C, H/2, W/2
        x = x.flatten(2).transpose(1, 2)   # B*4, C, H/2, W/2 -> B*4, C, S -> B*4, S, C
        x = self.block(x)
        x = self.layer_norm(x)
        x = x.transpose(1, 2).reshape(B, 4, C, H//2, W//2) # B*4, S, C -> B*4, C, S -> B, 4, C, H/2, W/2
        h1 = x[:, 0]
        h2 = x[:, 1]
        h3 = x[:, 2]
        h4 = x[:, 3]
        v1 = torch.cat([h1, h2], dim=3)
        v1 = v1.reshape(B, C, H//2, 2, W//2)
        v1 = v1.transpose(3, 4)
        v1 = v1.reshape(B, C, H//2, W)
        v2 = torch.cat([h3, h4], dim=3)
        v2 = v2.reshape(B, C, H//2, 2, W//2)
        v2 = v2.transpose(3, 4)
        v2 = v2.reshape(B, C, H//2, W)
        v = torch.cat([v1, v2], dim=2)
        v = v.reshape(B, C, 2, H//2, W)
        v = v.transpose(2, 3)
        v = v.reshape(B, C, H, W) # B, C, 2, H/2, 2, W/2 -> B, C, H, W
        return v

    def weight_init(self):
        # trunc_normal_(self.position_embedding.weight, std=.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "init_weights"):
                m.init_weights()


class Attention(nn.Module):
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 apply_WMSA: bool = False,
                 apply_GMMSA: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        if apply_WMSA:
            self.wmsa = WMSA(dim, num_heads)
        else:
            self.wmsa = nn.Identity()
        if apply_GMMSA:
            self.gmmsa = GMMSA(dim, num_heads)
        else:
            self.gmmsa = nn.Identity()
    
    def forward(self, x: Tensor):
        B, C, H, W = x.shape
        if H % 2 != 0:
            H += 1
        if W % 2 != 0:
            W += 1
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.wmsa(x)
        x = self.gmmsa(x)
        return x