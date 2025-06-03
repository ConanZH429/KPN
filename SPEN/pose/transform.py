import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Dict

class PosTransform(nn.Module):
    def __init__(
            self,
            pos_type: str,
            **kwargs
    ):
        super().__init__()
        if pos_type == "DiscreteSpher":
            self.angle_stride = kwargs.get("angle_stride", 1)
            self.r_stride = kwargs.get("r_stride", 1)
            self.r_max = kwargs.get("r_max", 50)
            self.device = kwargs.get("device", "cpu")
            self.theta_len = int(90 // self.angle_stride + 1)
            self.phi_len = int(360 // self.angle_stride + 1)
            self.r_len = int(self.r_max // self.r_stride + 1)
            self.theta_range = torch.linspace(0, 90, self.theta_len, device=self.device, requires_grad=False)
            self.phi_range = torch.linspace(0, 360, self.phi_len, device=self.device, requires_grad=False) - 180
            self.r_range = torch.linspace(0, self.r_max, self.r_len, device=self.device, requires_grad=False)

    def transform(self, pos_dict: Dict[str, Tensor]):
        if "cart" in pos_dict:
            pos_dict["spher"] = self.cart2spher(pos_dict["cart"])
        elif "spher" in pos_dict:
            pos_dict["cart"] = self.spher2cart(pos_dict["spher"])
        elif "discrete_r" in pos_dict:
            pos_dict["spher"] = self.discrete_spher2spher(
                pos_dict["discrete_r"],
                pos_dict["discrete_theta"],
                pos_dict["discrete_phi"]
            )
            pos_dict["cart"] = self.spher2cart(pos_dict["spher"])
        return pos_dict
    
    def spher2cart(self, spher: Tensor):
        r, theta, phi = spher.unbind(1)

        z = r * torch.cos(theta)
        r_sin_theta = r * torch.sin(theta)
        x = r_sin_theta * torch.cos(phi)
        y = r_sin_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=1)

    def cart2spher(self, pos: Tensor):
        x, y, z = pos.unbind(1)
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        phi = torch.atan2(y, x)
        spher = torch.stack([r, theta, phi], dim=1)
    
    def discrete_spher2spher(self, discrete_r: Tensor, discrete_theta: Tensor, discrete_phi: Tensor):
        discrete_r = F.softmax(discrete_r, dim=-1)
        discrete_theta = F.softmax(discrete_theta, dim=-1)
        discrete_phi = F.softmax(discrete_phi, dim=-1)

        r = torch.sum(discrete_r * self.r_range, dim=1)
        theta = torch.deg2rad(torch.sum(discrete_theta * self.theta_range, dim=1))
        phi = torch.deg2rad(torch.sum(discrete_phi * self.phi_range, dim=1))

        return torch.stack([r, theta, phi], dim=1)


class OriTransform(nn.Module):
    def __init__(
            self,
            ori_type: str,
            **kwargs,
    ):
        super().__init__()
        if ori_type == "DiscreteEuler":
            self.stride = kwargs.get("stride", 1)
            self.device = kwargs.get("device", "cpu")
            self.yaw_len = int(360 // self.stride + 1)
            self.pitch_len = int(180 // self.stride + 1)
            self.roll_len = int(360 // self.stride + 1)       # -180 ~ 180
            self.yaw_range = torch.linspace(0, 360, self.yaw_len, device=self.device, requires_grad=False) - 180        # -180 ~ 180
            self.pitch_range = torch.linspace(0, 180, self.pitch_len, device=self.device, requires_grad=False) - 90        # -90 ~ 90
            self.roll_range = torch.linspace(0, 360, self.roll_len, device=self.device, requires_grad=False) - 180        # -180 ~ 180
    
    def transform(self, ori_dict: Dict[str, Tensor]):
        if "quat" in ori_dict:
            ori_dict["euler"] = self.quat2euler(ori_dict["quat"])
        elif "euler" in ori_dict:
            ori_dict["quat"] = self.euler2quat(ori_dict["euler"])
        elif "discrete_yaw" in ori_dict:
            ori_dict["euler"] = self.discrete_euler2euler(
                ori_dict["discrete_yaw"],
                ori_dict["discrete_pitch"],
                ori_dict["discrete_roll"]
            )
            ori_dict["quat"] = self.euler2quat(ori_dict["euler"])
        return ori_dict

    def euler2quat(self, euler: Tensor):
        yaw, pitch, roll = euler.unbind(1)
        half_yaw = yaw * 0.5
        half_pitch = pitch * 0.5
        half_roll = roll * 0.5

        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        cp = torch.cos(half_pitch)
        sp = torch.sin(half_pitch)
        cr = torch.cos(half_roll)
        sr = torch.sin(half_roll)
        
        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * sp * cr + sy * cp * sr
        q2 = sy * cp * cr - cy * sp * sr
        q3 = -sy * sp * cr + cy * cp * sr

        return torch.stack([q0, q1, q2, q3], dim=1)
    
    def quat2euler(self, quat: Tensor):
        q0, q1, q2, q3 = quat.unbind(1)
        q1_2 = q1**2
        q2_2 = q2**2
        q3_2 = q3**2
        yaw = torch.atan2(
            2*(q0*q2 + q1*q3),
            1 - 2*(q2_2+q1_2)
        )
        pitch = torch.asin(2*(q0*q1 - q2*q3))
        roll = torch.atan2(
            2*(q0*q3 + q1*q2),
            1 - 2*(q1_2+q3_2)
        )
        return torch.stack([yaw, pitch, roll], dim=1)
    
    def discrete_euler2euler(self, discrete_yaw: Tensor, discrete_pitch: Tensor, discrete_roll: Tensor):
        discrete_yaw = F.softmax(discrete_yaw, dim=-1)
        discrete_pitch = F.softmax(discrete_pitch, dim=-1)
        discrete_roll = F.softmax(discrete_roll, dim=-1)

        yaw = torch.sum(discrete_yaw * self.yaw_range, dim=1)
        pitch = torch.sum(discrete_pitch * self.pitch_range, dim=1)
        roll = torch.sum(discrete_roll * self.roll_range, dim=1)

        return torch.deg2rad(torch.stack([yaw, pitch, roll], dim=1))