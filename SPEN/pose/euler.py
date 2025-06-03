import torch
import math
import numba
from torch import Tensor

import numpy as np

from scipy.spatial.transform import Rotation as R

@numba.njit
def quat2euler(q):
    q0, q1, q2, q3 = q
    q1_2 = q1**2
    q2_2 = q2**2
    q3_2 = q3**2
    yaw = math.atan2(
        2*(q0*q2 + q1*q3),
        1 - 2*(q2_2+q1_2)
    )
    pitch = math.asin(2*(q0*q1 - q2*q3))
    roll = math.atan2(
        2*(q0*q3 + q1*q2),
        1 - 2*(q1_2+q3_2)
    )
    return np.array([yaw, pitch, roll])

class EulerEncoder():
    def __init__(self, device: str = "cpu"):
        pass

    def encode(self, ori: np.ndarray) -> dict[str, np.ndarray]:
        """
        Encode the quaternion to Euler angle
        Args:
            ori (np.ndarray): the quaternion to encode
        
        Returns:
            dict[str, np.ndarray]: the encoded Euler angle
        """
        # rotation = R.from_quat(ori, scalar_first=True)
        # euler = rotation.as_euler("YXZ", degrees=False)
        euler = quat2euler(ori)

        return {
            "euler": euler.astype(np.float32)
        }


class EulerDecoder():
    def __init__(self, device: str = "cuda"):
        self.device = device

    def decode_batch(self, ori_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the Euler angle to quaternion
        Args:
            ori_pre_dict (dict[str, Tensor]): the Euler angle to decode
        
        Returns:
            Tensor: the decoded quaternion
        """
        euler = ori_pre_dict["euler"]
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