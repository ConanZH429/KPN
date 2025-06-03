import torch
import math
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from typing import Union, List, Literal, Dict

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
        self.loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, pre, label):
        return self.loss(F.log_softmax(pre, dim=1), label)


class CartLoss(nn.Module):

    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: Dict[str, float] = {"x": 1, "y": 1, "z": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"x": None, "y": None, "z": None},
            **kwargs
    ):
        super().__init__()
        self.x_loss = CartLoss.loss_dict[loss_type](**kwargs)
        self.y_loss = CartLoss.loss_dict[loss_type](**kwargs)
        self.z_loss = CartLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "x_beta": Beta(beta["x"], weight_strategy["x"], **kwargs),
            "y_beta": Beta(beta["y"], weight_strategy["y"], **kwargs),
            "z_beta": Beta(beta["z"], weight_strategy["z"], **kwargs),
        }

    def forward(
            self,
            pos_pre_dict: dict[str, Tensor],
            pos_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["x_beta"].step(now_epoch=now_epoch)
        self.beta_dict["y_beta"].step(now_epoch=now_epoch)
        self.beta_dict["z_beta"].step(now_epoch=now_epoch)
        x_pre, y_pre, z_pre = pos_pre_dict["cart"].split([1, 1, 1], dim=1)
        x_label, y_label, z_label = pos_label_dict["cart"].split([1, 1, 1], dim=1)
        return {
            "x_loss": self.x_loss(x_pre, x_label) * self.beta_dict["x_beta"].beta,
            "y_loss": self.y_loss(y_pre, y_label) * self.beta_dict["y_beta"].beta,
            "z_loss": self.z_loss(z_pre, z_label) * self.beta_dict["z_beta"].beta
        }


class SpherLoss(nn.Module):

    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: Dict[str, float] = {"r": 1, "theta": 1, "phi": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"r": None, "theta": None, "phi": None},
            **kwargs
    ):
        super().__init__()
        self.r_loss = SpherLoss.loss_dict[loss_type](**kwargs)
        self.theta_loss = SpherLoss.loss_dict[loss_type](**kwargs)
        self.phi_loss = SpherLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "r_beta": Beta(beta["r"], weight_strategy["r"], **kwargs),
            "theta_beta": Beta(beta["theta"], weight_strategy["theta"], **kwargs),
            "phi_beta": Beta(beta["phi"], weight_strategy["phi"], **kwargs),
        }
    
    def forward(
            self,
            pos_pre_dict: dict[str, Tensor],
            pos_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["r_beta"].step(now_epoch=now_epoch)
        self.beta_dict["theta_beta"].step(now_epoch=now_epoch)
        self.beta_dict["phi_beta"].step(now_epoch=now_epoch)
        r_pre, theta_pre, phi_pre = pos_pre_dict["spher"].split([1, 1, 1], dim=1)
        r_label, theta_label, phi_label = pos_label_dict["spher"].split([1, 1, 1], dim=1)
        return {
            "r_loss": self.r_loss(r_pre, r_label) * self.beta_dict["r_beta"].beta,
            "theta_loss": self.theta_loss(theta_pre, theta_label) * self.beta_dict["theta_beta"].beta,
            "phi_loss": self.phi_loss(phi_pre, phi_label) * self.beta_dict["phi_beta"].beta
        }
    

class DiscreteSpherLoss(nn.Module):

    loss_dict = {
        "CE": nn.CrossEntropyLoss,
        "KL": KLLoss,
    }

    def __init__(
            self,
            loss_type: str,
            beta: Dict[str, float] = {"discrete_r": 1, "discrete_theta": 1, "discrete_phi": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"discrete_r": None, "discrete_theta": None, "discrete_phi": None},
            **kwargs
    ):
        super().__init__()
        self.discrete_r_loss = DiscreteSpherLoss.loss_dict[loss_type](**kwargs)
        self.discrete_theta_loss = DiscreteSpherLoss.loss_dict[loss_type](**kwargs)
        self.discrete_phi_loss = DiscreteSpherLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "discrete_r_beta": Beta(beta["discrete_r"], weight_strategy["discrete_r"], **kwargs),
            "discrete_theta_beta": Beta(beta["discrete_theta"], weight_strategy["discrete_theta"], **kwargs),
            "discrete_phi_beta": Beta(beta["discrete_phi"], weight_strategy["discrete_phi"], **kwargs),
        }

    def forward(
            self,
            pos_pre_dict: dict[str, Tensor],
            pos_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["discrete_r_beta"].step(now_epoch=now_epoch)
        self.beta_dict["discrete_theta_beta"].step(now_epoch=now_epoch)
        self.beta_dict["discrete_phi_beta"].step(now_epoch=now_epoch)
        discrete_r_pre = pos_pre_dict["discrete_r"]
        discrete_theta_pre = pos_pre_dict["discrete_theta"]
        discrete_phi_pre = pos_pre_dict["discrete_phi"]
        discrete_r_label = pos_label_dict["discrete_r"]
        discrete_theta_label = pos_label_dict["discrete_theta"]
        discrete_phi_label = pos_label_dict["discrete_phi"]
        return {
            "discrete_r_loss": self.discrete_r_loss(discrete_r_pre, discrete_r_label) * self.beta_dict["discrete_r_beta"].beta,
            "discrete_theta_loss": self.discrete_theta_loss(discrete_theta_pre, discrete_theta_label) * self.beta_dict["discrete_theta_beta"].beta,
            "discrete_phi_loss": self.discrete_phi_loss(discrete_phi_pre, discrete_phi_label) * self.beta_dict["discrete_phi_beta"].beta
        }


# QuatLoss
class CosSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosSimilarityLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(torch.arccos(ori_inner_dot))


class CosDistanceLoss(nn.Module):
    def __init__(self):
        super(CosDistanceLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(1 - ori_inner_dot)


class ExpCosDistanceLossLoss(nn.Module):
    def __init__(self):
        super(ExpCosDistanceLossLoss, self).__init__()
    
    def forward(self, ori_pre: Tensor, ori_label: Tensor):
        ori_inner_dot = torch.abs(torch.sum(ori_pre * ori_label, dim=1))
        ori_inner_dot = torch.clamp(ori_inner_dot, max=1.0, min=-1.0)
        return torch.mean(torch.exp(1 - ori_inner_dot))


class QuatLoss(nn.Module):

    loss_dict = {
        "Cos": CosSimilarityLoss,
        "CosDistance": CosDistanceLoss,
        "ExpCos": ExpCosDistanceLossLoss,
    }
    
    def __init__(
            self,
            loss_type: str = "Cos",
            beta: float = 1,
            weight_strategy: WeightStrategy = None,
            **kwargs
    ):
        super().__init__()
        self.loss = QuatLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "quat_beta": Beta(beta, weight_strategy, **kwargs),
        }
    
    def forward(
            self,
            ori_pre_dict: dict[str, Tensor],
            ori_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["quat_beta"].step(now_epoch=now_epoch)
        quat_pre = ori_pre_dict["quat"]
        quat_label = ori_label_dict["quat"]
        return {
            "quat_loss": self.loss(quat_pre, quat_label) * self.beta_dict["quat_beta"].beta
        }


class EulerLoss(nn.Module):

    loss_dict = {
        "L1": nn.L1Loss,
        "L2": nn.MSELoss,
        "SmoothL1": nn.SmoothL1Loss,
    }

    def __init__(
            self,
            loss_type: str = "L1",
            beta: Dict[str, float] = {"yaw": 1, "pitch": 1, "roll": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"yaw": None, "pitch": None, "roll": None},
            **kwargs
    ):
        super().__init__()
        self.yaw_loss = EulerLoss.loss_dict[loss_type](**kwargs)
        self.pitch_loss = EulerLoss.loss_dict[loss_type](**kwargs)
        self.roll_loss = EulerLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "yaw_beta": Beta(beta["yaw"], weight_strategy["yaw"], **kwargs),
            "pitch_beta": Beta(beta["pitch"], weight_strategy["pitch"], **kwargs),
            "roll_beta": Beta(beta["roll"], weight_strategy["roll"], **kwargs),
        }
    
    def forward(
            self,
            ori_pre_dict: dict[str, Tensor],
            ori_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["yaw_beta"].step(now_epoch=now_epoch)
        self.beta_dict["pitch_beta"].step(now_epoch=now_epoch)
        self.beta_dict["roll_beta"].step(now_epoch=now_epoch)
        yaw_pre, pitch_pre, roll_pre = ori_pre_dict["euler"].split([1, 1, 1], dim=1)
        yaw_label, pitch_label, roll_label = ori_label_dict["euler"].split([1, 1, 1], dim=1)
        return {
            "yaw_loss": self.yaw_loss(yaw_pre, yaw_label) * self.beta_dict["yaw_beta"].beta,
            "pitch_loss": self.pitch_loss(pitch_pre, pitch_label) * self.beta_dict["pitch_beta"].beta,
            "roll_loss": self.roll_loss(roll_pre, roll_label) * self.beta_dict["roll_beta"].beta
        }
        

class DiscreteEulerLoss(nn.Module):

    loss_dict = {
        "CE": nn.CrossEntropyLoss,
        "KL": KLLoss,
    }

    def __init__(
            self,
            loss_type: str = "CE",
            beta: Dict[str, float] = {"discrete_yaw": 1, "discrete_pitch": 1, "discrete_roll": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"discrete_yaw": None, "discrete_pitch": None, "discrete_roll": None},
            **kwargs
    ):
        super().__init__()
        self.discrete_yaw_loss = DiscreteEulerLoss.loss_dict[loss_type](**kwargs)
        self.discrete_pitch_loss = DiscreteEulerLoss.loss_dict[loss_type](**kwargs)
        self.discrete_roll_loss = DiscreteEulerLoss.loss_dict[loss_type](**kwargs)
        self.beta_dict = {
            "discrete_yaw_beta": Beta(beta["discrete_yaw"], weight_strategy["discrete_yaw"], **kwargs),
            "discrete_pitch_beta": Beta(beta["discrete_pitch"], weight_strategy["discrete_pitch"], **kwargs),
            "discrete_roll_beta": Beta(beta["discrete_roll"], weight_strategy["discrete_roll"], **kwargs),
        }
    
    def forward(
            self,
            ori_pre_dict: dict[str, Tensor],
            ori_label_dict: dict[str, Tensor],
            **kwargs
    ):
        now_epoch = kwargs.get("now_epoch", None)
        self.beta_dict["discrete_yaw_beta"].step(now_epoch=now_epoch)
        self.beta_dict["discrete_pitch_beta"].step(now_epoch=now_epoch)
        self.beta_dict["discrete_roll_beta"].step(now_epoch=now_epoch)
        discrete_yaw_pre = ori_pre_dict["discrete_yaw"]
        discrete_pitch_pre = ori_pre_dict["discrete_pitch"]
        discrete_roll_pre = ori_pre_dict["discrete_roll"]
        discrete_yaw_label = ori_label_dict["discrete_yaw"]
        discrete_pitch_label = ori_label_dict["discrete_pitch"]
        discrete_roll_label = ori_label_dict["discrete_roll"]
        return {
            "discrete_yaw_loss": self.discrete_yaw_loss(discrete_yaw_pre, discrete_yaw_label) * self.beta_dict["discrete_yaw_beta"].beta,
            "discrete_pitch_loss": self.discrete_pitch_loss(discrete_pitch_pre, discrete_pitch_label) * self.beta_dict["discrete_pitch_beta"].beta,
            "discrete_roll_loss": self.discrete_roll_loss(discrete_roll_pre, discrete_roll_label) * self.beta_dict["discrete_roll_beta"].beta
        }


class PosLossFactory:
    def __init__(self):
        pass

    def create_pos_loss(
            self,
            pos_type: str,
            loss_type: str,
            beta: Dict[str, float] = {"x": 1, "y": 1, "z": 1},
            weight_strategy: Dict[str, WeightStrategy] = {"x": None, "y": None, "z": None},
            **kwargs
    ):
        if pos_type == "Cart":
            return CartLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
        elif pos_type == "Spher":
            return SpherLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
        elif pos_type == "DiscreteSpher":
            return DiscreteSpherLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)


class OriLossFactory:
    def __init__(self):
        pass

    def create_ori_loss(
            self,
            ori_type: str,
            loss_type: str,
            beta: Union[float, Dict[str, float]] = 1,
            weight_strategy: WeightStrategy = None,
            **kwargs
    ):
        if ori_type == "Quat":
            return QuatLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
        elif ori_type == "Euler":
            return EulerLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)
        elif ori_type == "DiscreteEuler":
            return DiscreteEulerLoss(loss_type, beta=beta, weight_strategy=weight_strategy, **kwargs)