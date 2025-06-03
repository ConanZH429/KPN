import torch
import numba
import math
from torch import Tensor
import torch.nn.functional as F

import numpy as np

from scipy.spatial.transform import Rotation as R

class DiscreteEuler():
    """
    discrete Euler angle probability distribution
    """

    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device to use
        """
        self.stride = stride
        self.device = device
        self.yaw_len = int(360 // stride + 1)
        self.pitch_len = int(180 // stride + 1)
        self.roll_len = int(360 // stride + 1)
        self.yaw_range = torch.linspace(0, 360, self.yaw_len, device=device, requires_grad=False) - 180        # -180 ~ 180
        self.pitch_range = torch.linspace(0, 180, self.pitch_len, device=device, requires_grad=False) - 90        # -90 ~ 90
        self.roll_range = torch.linspace(0, 360, self.roll_len, device=device, requires_grad=False) - 180        # -180 ~ 180
        self.yaw_index_dict = {int(yaw // stride): i for i, yaw in enumerate(self.yaw_range)}
        self.pitch_index_dict = {int(pitch // stride): i for i, pitch in enumerate(self.pitch_range)}
        self.roll_index_dict = {int(roll // stride): i for i, roll in enumerate(self.roll_range)}


@numba.njit
def quat2euler(q):
    rad2deg = 180 / math.pi
    q0, q1, q2, q3 = q
    q1_2 = q1**2
    q2_2 = q2**2
    q3_2 = q3**2
    yaw = math.atan2(
        2*(q0*q2 + q1*q3),
        1 - 2*(q2_2+q1_2)
    ) * rad2deg
    pitch = math.asin(2*(q0*q1 - q2*q3)) * rad2deg
    roll = math.atan2(
        2*(q0*q3 + q1*q2),
        1 - 2*(q1_2+q3_2)
    ) * rad2deg
    return yaw, pitch, roll


class DiscreteEulerEncoder(DiscreteEuler):
    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device
        """
        super().__init__(stride, device)

    def _encode_ori(self, angle: float, angle_len: int, index_dict: dict):
        angle_encode = np.zeros(angle_len, dtype=np.float32)

        mean = angle / self.stride
        l = int(np.floor(mean))
        r = int(np.ceil(mean))
        li = index_dict[l]
        ri = index_dict[r]
        if l == r:
            angle_encode[li] = 1
        else:
            angle_encode[li] = r - mean
            angle_encode[ri] = mean - l
        
        return angle_encode

    def encode(self, ori: np.ndarray) -> dict[str, np.ndarray]:
        """
        Encode the quaternion angle to the probability distribution
        Args:
            ori (np.ndarray): the Euler angle in quaternion format
        
        Returns:
            encode (dict[str, np.ndarray]): the probability distribution of yaw, pitch, and roll
        """
        yaw, pitch, roll = quat2euler(ori)
        
        discrete_yaw = self._encode_ori(yaw, self.yaw_len, self.yaw_index_dict)
        discrete_pitch = self._encode_ori(pitch, self.pitch_len, self.pitch_index_dict)
        discrete_roll = self._encode_ori(roll, self.roll_len, self.roll_index_dict)

        return {
            "discrete_yaw": discrete_yaw,
            "discrete_pitch": discrete_pitch,
            "discrete_roll": discrete_roll
        }


class DiscreteEulerDecoder(DiscreteEuler):
    def __init__(self, stride: int, device: str = "cuda"):
        """
        Args:
            stride (int): the stride of the grid
            device (str): the device
        """
        super().__init__(stride, device)
    
    def decode_batch(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the probability distribution to the Euler angle in batch
        
        Args:
            ori_pre_dict (dict[str, Tensor]): the probability distributions for yaw, pitch, and roll
        
        Returns:
            ori (Tensor): the Euler angle in quaternion format
        """
        
        discrete_yaw = F.softmax(ori_pre_dict["discrete_yaw"], dim=-1)
        discrete_pitch = F.softmax(ori_pre_dict["discrete_pitch"], dim=-1)
        discrete_roll = F.softmax(ori_pre_dict["discrete_roll"], dim=-1)

        yaw = torch.sum(discrete_yaw * self.yaw_range, dim=1)
        pitch = torch.sum(discrete_pitch * self.pitch_range, dim=1)
        roll = torch.sum(discrete_roll * self.roll_range, dim=1)

        half_yaw = torch.deg2rad(yaw * 0.5)
        half_pitch = torch.deg2rad(pitch * 0.5)
        half_roll = torch.deg2rad(roll * 0.5)
        
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