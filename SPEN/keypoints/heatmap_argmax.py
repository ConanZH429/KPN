import torch
import numpy as np

from torch import Tensor

from ..utils import Camera

from typing import Union, Tuple

class HeatmapArgmax:
    def __init__(self,):
        pass

class HeatmapArgmaxEncoder(HeatmapArgmax):
    def __init__(
            self,
            keypoints_num: int,
            input_image_shape: Tuple[int, int] = (480, 768),
            heatmap_ratio: float = 1/4,
            sigma: float = 3.0,
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
        self.sigma = sigma
        self.kernel_radius = int(3 * sigma)
        self.kernel_size = 2 * self.kernel_radius + 1
        self.kernel = self._create_gaussian_kernel(self.kernel_size, sigma)

    def _create_gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:
        """Create a Gaussian kernel."""
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        for x in range(size):
            for y in range(size):
                kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
        return kernel

    def encode(
            self,
            keypoints: np.ndarray,
    ):
        """
        keypoints: (N, 2)
        """
        heatmap = np.zeros(
            (self.keypoints_num, *self.heatmap_shape),
            dtype=np.float32
        )
        heatmap_keypoints = np.round(keypoints * self.heatmap_ratio, decimals=0).astype(np.int32)    # 关键点在热图中的位置
        for keypoint_idx in range(self.keypoints_num):
            keypoint = heatmap_keypoints[keypoint_idx]
            x, y = keypoint
            if 0 <= x <= self.heatmap_shape[1]-1 and 0 <= y <= self.heatmap_shape[0]-1:
                # 高斯核有效区域，高斯核坐标系
                g_x = ( max(0, self.kernel_radius - x), min(self.kernel_radius, self.heatmap_shape[1]-1-x) + self.kernel_radius + 1 )
                g_y = ( max(0, self.kernel_radius - y), min(self.kernel_radius, self.heatmap_shape[0]-1-y) + self.kernel_radius + 1 )
                # 热图有效区域，热图坐标系
                h_x = ( max(0, x - self.kernel_radius), min(x+self.kernel_radius+1, self.heatmap_shape[1]) )
                h_y = ( max(0, y - self.kernel_radius), min(y+self.kernel_radius+1, self.heatmap_shape[0]) )
                heatmap[keypoint_idx, h_y[0]:h_y[1], h_x[0]:h_x[1]] = self.kernel[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return heatmap


class HeatmapArgmaxDecoder(HeatmapArgmax):
    def __init__(
            self,
            heatmap_ratio: float = 1/4,
            **kwargs,
        ):
        super().__init__()
        self.heatmap_ratio = heatmap_ratio
    
    def decode(
            self,
            heatmap: Tensor
    ):
        """
        Heatmap: (B, N, H, W)
        """
        B, N, H, W = heatmap.shape
        heatmap = heatmap.view(B, N, -1)    # B, N, H*W
        max_indices = heatmap.argmax(dim=-1)    # B, N
        y = max_indices // W        # B, N
        x = max_indices % W         # B, N
        # Scale back to original image shape
        x = x.float() / self.heatmap_ratio
        y = y.float() / self.heatmap_ratio
        return torch.stack((x, y), dim=-1)  # B, N, 2