import torch
from torch import Tensor
import numpy as np
import math
import torch.nn.functional as F

class DiscreteSpher():
    """
    discrete spherical coordinate probability distribution
    """

    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius            
            r_max (int): maximum radius value
            device (str): the device to use
        """
        self.angle_stride = angle_stride
        self.r_stride = r_stride
        self.r_max = r_max
        self.device = device
        self.theta_len = int(90 // angle_stride + 1)
        self.phi_len = int(360 // angle_stride + 1)
        self.r_len = int(r_max // r_stride + 1)
        self.theta_range = torch.linspace(0, 90, self.theta_len, device=device, requires_grad=False)
        self.phi_range = torch.linspace(0, 360, self.phi_len, device=device, requires_grad=False) - 180
        self.r_range = torch.linspace(0, r_max, self.r_len, device=device, requires_grad=False)
        self.theta_index_dict = {int(theta // angle_stride): i for i, theta in enumerate(self.theta_range)}
        self.phi_index_dict = {int(phi // angle_stride): i for i, phi in enumerate(self.phi_range)}
        self.r_index_dict = {int(r // r_stride): i for i, r in enumerate(self.r_range)}
    

class DiscreteSpherEncoder(DiscreteSpher):
    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius
            r_max (int): maximum radius value
            device (str): the device
        """
        super().__init__(angle_stride, r_stride, r_max, device)
    
    def _encode_pos(self, x: float, x_len: int, x_stride: int, index_dict: dict):
        x_encode = np.zeros(x_len, dtype=np.float32)

        mean = x / x_stride
        l, r = int(np.floor(mean)), int(np.ceil(mean))
        if l == r:
            x_encode[index_dict[l]] = 1
        else:
            x_encode[index_dict[l]] = (r - mean) / (r - l)
            x_encode[index_dict[r]] = (mean - l) / (r - l)
        
        return x_encode

    def encode(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        """
        encode the position to discrete spherical coordinate probability distribution

        Args:
            pos (np.ndarray): the position in the format of (x, y, z)

        Returns:
            dict[str, np.ndarray]: the encoded position
        """
        x, y, z = pos
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z / r) * 180 / math.pi
        phi = math.atan2(y, x) * 180 / math.pi
        discrete_r = self._encode_pos(r, self.r_len, self.r_stride, self.r_index_dict)
        discrete_theta = self._encode_pos(theta, self.theta_len, self.angle_stride, self.theta_index_dict)
        discrete_phi = self._encode_pos(phi, self.phi_len, self.angle_stride, self.phi_index_dict)
        return {
            "discrete_r": discrete_r,
            "discrete_theta": discrete_theta,
            "discrete_phi": discrete_phi
        }


class DiscreteSpherDecoder(DiscreteSpher):
    def __init__(self, angle_stride: int, r_stride: int, r_max: int, device: str = "cuda"):
        """
        Args:
            angle_stride (int): the stride of the grid in polar and azimuthal angle
            r_stride (int): the stride of the grid in radius
            r_max (int): maximum radius value
            device (str): the device
        """
        super().__init__(angle_stride, r_stride, r_max, device)


    def decode_batch(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        decode the discrete spherical coordinate probability distribution to position

        Args:
            pos_pre_dict (dict[str, Tensor]): the probability distributions for radius, polar angle, and azimuthal angle

        Returns:
            np.ndarray: the decoded position in the format of (x, y, z)
        """
        discrete_r = F.softmax(pos_pre_dict["discrete_r"], dim=-1)
        discrete_theta = F.softmax(pos_pre_dict["discrete_theta"], dim=-1)
        discrete_phi = F.softmax(pos_pre_dict["discrete_phi"], dim=-1)

        r = torch.sum(discrete_r * self.r_range, dim=1)
        theta = torch.sum(discrete_theta * self.theta_range, dim=1)
        phi = torch.sum(discrete_phi * self.phi_range, dim=1)
        theta = torch.deg2rad(theta)
        phi = torch.deg2rad(phi)

        z = r * torch.cos(theta)
        r_sin_theta = r * torch.sin(theta)
        x = r_sin_theta * torch.cos(phi)
        y = r_sin_theta * torch.sin(phi)

        return torch.stack([x, y, z], dim=1)