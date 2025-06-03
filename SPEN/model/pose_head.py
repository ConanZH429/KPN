from .blocks import *

from typing import Dict, Any, List

# head
class CartHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        in_channels = in_channels[0]
        self.cart_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, pos_feature: Tensor):
        cart = self.cart_fc(pos_feature)
        return {
            "cart": cart
        }


class SpherHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        in_channels = in_channels[0]
        self.spher_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        spher = self.spher_fc(x)
        return {
            "spher": spher
        }


class DiscreteSpherHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        angle_stride = kwargs.get("angle_stride", None)
        r_stride = kwargs.get("r_stride", None)
        r_max = kwargs.get("r_max", None)
        r_dim = int(r_max // r_stride + 1)
        theta_dim = int(90 // angle_stride + 1)
        phi_dim = int(360 // angle_stride + 1)
        if len(in_channels) == 1:
            r_in_channels = theta_in_channels = phi_in_channels = in_channels[0]
        elif len(in_channels) == 3:
            r_in_channels, theta_in_channels, phi_in_channels = in_channels
        self.r_fc = nn.Linear(r_in_channels, r_dim)
        self.theta_fc = nn.Linear(theta_in_channels, theta_dim)
        self.phi_fc = nn.Linear(phi_in_channels, phi_dim)

    def forward(self, pos_feature: Tensor):
        if pos_feature.dim() == 3:
            r_feature, theta_feature, phi_feature = pos_feature[:, :, 0], pos_feature[:, :, 1], pos_feature[:, :, 2]
        elif pos_feature.dim() == 2:
            r_feature = theta_feature = phi_feature = pos_feature
        discrete_r = self.r_fc(r_feature)
        discrete_theta = self.theta_fc(theta_feature)
        discrete_phi = self.phi_fc(phi_feature)
        return {
            "discrete_r": discrete_r,
            "discrete_theta": discrete_theta,
            "discrete_phi": discrete_phi
        }


class QuatHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        in_channels = in_channels[0]
        self.quat_fc = nn.Linear(in_channels, 4, bias=False)
    
    def forward(self, x: Tensor):
        quat = self.quat_fc(x)
        quat = F.normalize(quat, p=2, dim=1)
        return {
            "quat": quat
        }


class EulerHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        in_channels = in_channels[0]
        self.euler_fc = nn.Linear(in_channels, 3, bias=False)
    
    def forward(self, x: Tensor):
        euler = self.euler_fc(x)
        return {
            "euler": euler
        }


class DiscreteEulerHead(nn.Module):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        stride = kwargs.get("stride", None)
        yaw_dim = 360 // stride + 1
        pitch_dim = 180 // stride + 1
        roll_dim = 360 // stride + 1
        if len(in_channels) == 1:
            yaw_in_channels = pitch_in_channels = roll_in_channels = in_channels[0]
        elif len(in_channels) == 3:
            yaw_in_channels, pitch_in_channels, roll_in_channels = in_channels
        self.yaw_fc = nn.Linear(yaw_in_channels, yaw_dim)
        self.pitch_fc = nn.Linear(pitch_in_channels, pitch_dim)
        self.roll_fc = nn.Linear(roll_in_channels, roll_dim)

    def forward(self, ori_feature: Tensor):
        if ori_feature.dim() == 3:
            yaw_feature, pitch_feature, roll_feature = ori_feature[:, :, 0], ori_feature[:, :, 1], ori_feature[:, :, 2]
        elif ori_feature.dim() == 2:
            yaw_feature = pitch_feature = roll_feature = ori_feature
        discrete_yaw = self.yaw_fc(yaw_feature)
        discrete_pitch = self.pitch_fc(pitch_feature)
        discrete_roll = self.roll_fc(roll_feature)
        return {
            "discrete_yaw": discrete_yaw,
            "discrete_pitch": discrete_pitch,
            "discrete_roll": discrete_roll
        }


class PoseHeadFactory:

    pos_head_dict = {
        "Cart": CartHead,
        "Spher": SpherHead,
        "DiscreteSpher": DiscreteSpherHead,
    }

    ori_head_dict = {
        "Quat": QuatHead,
        "Euler": EulerHead,
        "DiscreteEuler": DiscreteEulerHead
    }

    def __init__(self):
        pass
    
    def create_pose_head(
            self,
            pos_type: str,
            pos_args: Dict[str, Any],
            ori_type: str,
            ori_args: Dict[str, Any],
            pos_feature_dims: List[int],
            ori_feature_dims: List[int],
    ):
        PosHeadClass = PoseHeadFactory.pos_head_dict[pos_type]
        OriHeadClass = PoseHeadFactory.ori_head_dict[ori_type]
        pos_head_model = PosHeadClass(pos_feature_dims, **pos_args)
        ori_head_model = OriHeadClass(ori_feature_dims, **ori_args)
        return pos_head_model, ori_head_model