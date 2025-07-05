import torch
import numpy as np
from rich import print
from SPEN.keypoints import HeatmapArgmaxEncoder, HeatmapArgmaxDecoder
from SPEN.keypoints import HeatmapDistributionEncoder, HeatmapDistributionDecoder

def print_heatmap(heatmap):
    for row in heatmap:
        print("  ".join(f"{value:.4f}" for value in row))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

key_points_num = 5
image_shape = (40, 40)
shape_ratio = 1 / 4
sigma = 1.0
keypoints = np.array([
    [9.50, 2.00],
    [4.5, 5.7],
    [1.06, 2.33],
    [8.36, 7.08],
    [1.27, 7.21],
    [1.8, 2.00],
]) * 4
points_vis = np.array([1, 1, 1, 1, 1], dtype=np.bool)  # All keypoints are visible

print("Keypoints:")
print(keypoints)

heatmap_argmax_encoder = HeatmapArgmaxEncoder(
    keypoints_num=key_points_num,
    input_image_shape=image_shape,
    shape_ratio=shape_ratio,
    sigma=sigma
)
heatmap_argmax_decoder = HeatmapArgmaxDecoder(
    shape_ratio=shape_ratio,
)

heatmap = heatmap_argmax_encoder.encode(keypoints=keypoints)    # (5, 10, 10)
heatmap = torch.from_numpy(heatmap).float().unsqueeze(0)  # (1, 5, 10, 10)
keypoints_decode = heatmap_argmax_decoder.decode(heatmap)  # (1, 5, 2)
print("Heatmap Argmax Encoder:")
print(keypoints_decode)

heatmap_distribution_encoder = HeatmapDistributionEncoder(
    keypoints_num=key_points_num,
    input_image_shape=image_shape,
    ratio=shape_ratio
)

heatmap_distribution_decoder = HeatmapDistributionDecoder(
    input_image_shape=image_shape,
    ratio=shape_ratio
).to(device=device)

heatmap = heatmap_distribution_encoder.encode(keypoints=keypoints, points_vis=points_vis)  # (5, 10, 10)
heatmap = torch.from_numpy(heatmap).float().unsqueeze(0).to(device=device)  # (1, 5, 10, 10)
keypoints_decode = heatmap_distribution_decoder.decode(heatmap)  # (1, 5, 2)
print("Heatmap Distribution Decoder:")
print(keypoints_decode)