import torch
import math
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Literal, Dict, Optional

WeightStrategy = Literal[None, "BetaDecay"]

class Beta:
    def __init__(
            self,
            beta: float = 1,
            weight_strategy: WeightStrategy = None,
            **kwargs
    ):
        self.init_beta = beta        # 权重系数
        self.beta = beta
        self.weight_strategy = weight_strategy
        if weight_strategy == "CosDecay":
            self.max_iter = kwargs.get("max_iter", 400) # 最大epochs数
            self.min_ratio = kwargs.get("min_ratio", 0.1) # 最小权重比例
            self.step = self.cos_decay_step
            self.beta_ratio_list = [self.min_ratio + (1-self.min_ratio)*(1 + math.cos(math.pi * i / (self.max_iter - 1))) / 2
                                    for i in range(self.max_iter)]
    
    def step(self, *args, **kwargs):
        pass

    def cos_decay_step(self, now_epoch: int):
        if now_epoch > self.max_iter:
            ratio = self.min_ratio
        else:
            ratio = self.beta_ratio_list[now_epoch]
        self.beta = ratio * self.init_beta
        

@torch.compile
class KLLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = nn.KLDivLoss(reduction="none")
    
    def forward(self, pre, label):
        return self.loss(F.log_softmax(pre, dim=1), label).sum(dim=1)

class ArgmaxHeatmapLoss(nn.Module):
    
    loss_dict = {
        "L2": nn.MSELoss
    }

    def __init__(
            self,
            loss_type: str,
            beta: float = 1,
            weight_strategy: Union[WeightStrategy, None] = None,
            **kwargs,
    ):
        super().__init__()
        self.loss = ArgmaxHeatmapLoss.loss_dict[loss_type](**kwargs)
        self.beta = Beta(beta, weight_strategy)
    
    def forward(
            self,
            heatmap_pre: Tensor,        # (B, N, H, W)
            heatmap_label: Tensor,      # (B, N, H, W)
            points_vis: Tensor,             # (B, N) visibility mask
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta.step(now_epoch=now_epoch)
        loss_items = self.loss(heatmap_pre, heatmap_label)  # (B, N, H, W)
        loss_items = loss_items * points_vis.float().unsqueeze(-1).unsqueeze(-1)
        loss = loss_items.sum(dim=(1, 2, 3)).mean()     # 
        return loss * self.beta.beta


class DistributionHeatmapLoss(nn.Module):
    
    loss_dict = {
        "CE": nn.CrossEntropyLoss,
        "KL": KLLoss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: float = 1,
            weight_strategy: Union[WeightStrategy, None] = None,
            **kwargs,
    ):
        super().__init__()
        self.loss = DistributionHeatmapLoss.loss_dict[loss_type](**kwargs)
        self.beta = Beta(beta, weight_strategy)
    
    def forward(
        self,
        heatmap_pre: Tensor,        # (B, N, H, W)
        heatmap_label: Tensor,      # (B, N, H, W)
        points_vis: Tensor,         # (B, N) visibility mask
        **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta.step(now_epoch=now_epoch)
        B, N, H, W = heatmap_pre.shape
        pre = heatmap_pre.reshape(B*N, H*W)  # (B*N, H*W)
        label = heatmap_label.reshape(B*N, H*W)
        loss_item = self.loss(pre, label).reshape(B, N)   # (B*N,) -> (B, N)
        loss_item = loss_item * points_vis.float()
        loss = loss_item.sum(dim=1).mean()  # average over batch
        return loss * self.beta.beta


def get_keypoints_loss(
        keypoints_type: str,
        loss_type: str,
        beta: float = 1,
        weight_strategy: Union[WeightStrategy, None] = None,
        **kwargs,
):
    if keypoints_type == "heatmap_argmax":
        return ArgmaxHeatmapLoss(loss_type=loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
    elif keypoints_type == "heatmap_distribution":
        return DistributionHeatmapLoss(loss_type=loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
    else:
        raise ValueError(f"Unknown keypoints type: {keypoints_type}")


class SubLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
    
    def forward(
            self,
            uncertainty_pre: Tensor,        # (B, N)
            uncertainty_label: Tensor,      # (B, N)
            **kwargs
    ):
        B, N = uncertainty_pre.shape
        uncertainty_idx = torch.argsort(uncertainty_label, dim=1)   # (B, N) from small to large
        uncertainty = torch.gather(uncertainty_pre, 1, uncertainty_idx) # (B, N)
        loss = uncertainty[:, :-1] - uncertainty[:, 1:] # (B, N-1)
        loss = loss[loss > 0]  # only keep positive loss
        return torch.sum(loss) / B if loss.numel() > 0 else torch.tensor(0.0, device=uncertainty_pre.device)


class RankLoss(nn.Module):
    loss_dict = {
        "Sub": SubLoss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: float = 1,
            weight_strategy: Union[WeightStrategy, None] = None,
            **kwargs,
    ):
        super().__init__()
        self.loss = RankLoss.loss_dict[loss_type](**kwargs)
        self.beta = Beta(beta, weight_strategy)
    
    def forward(
            self,
            uncertainty_pre: Tensor,        # (B, N)
            uncertainty_label: Tensor,      # (B, N)
            points_vis: Tensor,             # (B, N) visibility mask
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta.step(now_epoch=now_epoch)
        return self.loss(uncertainty_pre, uncertainty_label) * self.beta.beta


class RatioLoss(nn.Module):
    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: float = 1,
            weight_strategy: Union[WeightStrategy, None] = None,
            **kwargs,
    ):
        super().__init__()
        self.loss = RatioLoss.loss_dict[loss_type](**kwargs)
        self.beta = Beta(beta, weight_strategy)
    
    def forward(
            self,
            uncertainty_pre: Tensor,        # (B, N)
            uncertainty_label: Tensor,      # (B, N)
            points_vis: Tensor,             # (B, N) visibility mask
            **kwargs):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta.step(now_epoch=now_epoch)
        loss_items = self.loss(uncertainty_pre, uncertainty_label)
        loss_items = loss_items * points_vis.float()
        loss = loss_items.sum(dim=1).mean()  # average over batch
        return loss * self.beta.beta


def get_uncertainty_loss(
        uncertainty_type: str,
        loss_type: str,
        beta: float = 1,
        weight_strategy: Union[WeightStrategy, None] = None,
        **kwargs,
):
    if uncertainty_type == "Rank":
        return RankLoss(loss_type=loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
    elif uncertainty_type == "Ratio":
        return RatioLoss(loss_type=loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
    else:
        raise ValueError(f"Unknown uncertainty type: {uncertainty_type}")