import torch
import math
from torch import Tensor

import numpy as np

class SpherEncoder():
    def __init__(self, device: str = "cpu"):
        pass

    def encode(self, pos: np.ndarray) -> dict[str, np.ndarray]:
        x, y, z = pos
        r = math.sqrt(x**2 + y**2 + z**2)
        theta = math.acos(z / r)
        phi = math.atan2(y, x)
        spher = np.array([r, theta, phi], dtype=np.float32)
        return {
            "spher": spher
        }
    

class SpherDecoder():
    def __init__(self, device: str = "cuda"):
        pass
    
    def decode_batch(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        """
        Decode the spherical coordinate to Cartesian coordinate
        Args:
            pos_pre_dict (dict[str, Tensor]): the spherical coordinate to decode
        """
        spher = pos_pre_dict["spher"]
        r, theta, phi = spher.unbind(1)

        z = r * torch.cos(theta)
        r_sin_theta = r * torch.sin(theta)
        x = r_sin_theta * torch.cos(phi)
        y = r_sin_theta * torch.sin(phi)
        return torch.stack([x, y, z], dim=1)
