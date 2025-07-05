import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor

from ..utils import Camera

from typing import Tuple

class HeatmapDistribution(nn.Module):
    def __init__(self,):
        super().__init__()


class HeatmapDistributionEncoder(HeatmapDistribution):
    def __init__(
            self,
            keypoints_num: int,
            input_image_shape: Tuple[int, int],
            heatmap_ratio: float = 1/2**5,
            **kwargs,
    ):
        super().__init__()
        self.keypoints_num = keypoints_num
        self.input_image_shape = input_image_shape
        self.heatmap_ratio = heatmap_ratio
        self.heatmap_shape = (
            int(input_image_shape[0] * heatmap_ratio),
            int(input_image_shape[1] * heatmap_ratio)
        )
    
    def _encode_coord(self, c: float, c_len: int) -> np.ndarray:
        c_encode = np.zeros(c_len, dtype=np.float32)
        
        l = int(np.floor(c))
        r = int(np.ceil(c))
        if l == r:
            c_encode[l] = 1
        else:
            c_encode[l] = r - c
            c_encode[r] = c - l
        return c_encode

    def _encode_keypoint_coord(self, x: float, y: float) -> np.ndarray:
        x_encode = self._encode_coord(x, self.heatmap_shape[1])
        y_encode = self._encode_coord(y, self.heatmap_shape[0])
        heatmap = y_encode.reshape(-1, 1) * x_encode
        return heatmap

    def encode(
            self, 
            keypoints: np.ndarray,
            points_vis: np.ndarray,
    ):
        """
        keypoints: (N, 2)
        """
        heatmap = np.zeros(
            (self.keypoints_num, *self.heatmap_shape),
            dtype=np.float32
        )
        heatmap_keypoints = keypoints * self.heatmap_ratio  # 关键点在热图中的位置
        for keypoint_idx in range(self.keypoints_num):
            keypoints = heatmap_keypoints[keypoint_idx]
            x, y = keypoints
            if points_vis[keypoint_idx]:
                heatmap[keypoint_idx] = self._encode_keypoint_coord(x, y)
        return heatmap


class HeatmapDistributionDecoder(HeatmapDistribution):
    def __init__(
            self,
            input_image_shape: Tuple[int, int],
            heatmap_ratio: float = 1/2**5,
            **kwargs,
    ):
        super().__init__()
        self.heatmap_ratio = heatmap_ratio
        self.input_image_shape = input_image_shape
        self.heatmap_shape = (
            int(input_image_shape[0] * heatmap_ratio),
            int(input_image_shape[1] * heatmap_ratio)
        )
        x_index = np.arange(self.heatmap_shape[1], dtype=np.float32)
        y_index = np.arange(self.heatmap_shape[0], dtype=np.float32)
        x_index, y_index = np.meshgrid(x_index, y_index, indexing='xy')
        self.register_buffer('x_index', torch.from_numpy(x_index).float())
        self.register_buffer('y_index', torch.from_numpy(y_index).float())
        
    
    def decode(
            self,
            keypoints_encode: Tensor,
    ):
        """
        keypoints_encode: (B, N, H, W)
        """
        B, N, H, W = keypoints_encode.shape
        keypoints_encode_softmax = F.softmax(keypoints_encode.reshape(B, N, -1), -1).reshape(B, N, H, W)
        # decode heatmap to keypoint coordinates
        x = torch.sum(self.x_index.view(1, 1, H, W) * keypoints_encode_softmax, dim=(2, 3))
        y = torch.sum(self.y_index.view(1, 1, H, W) * keypoints_encode_softmax, dim=(2, 3))
        # Scale back to original image shape
        x = x / self.heatmap_ratio
        y = y / self.heatmap_ratio
        return torch.stack((x, y), dim=-1)  # B, N, 2