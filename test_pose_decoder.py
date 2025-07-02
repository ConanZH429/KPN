import torch
import random
from SPEN.cfg import SPEEDplusConfig
from SPEN.module.utils import PoseDecoder, KeypointsDistanceCalculator
from SPEN.data import SPEEDplusTrainDataset
from torch.utils.data import DataLoader

config = SPEEDplusConfig()
config.cache = False
pose_decoder = PoseDecoder(
    image_size=config.image_size,
    points_world=config.keypoints,
    device="cpu"
)
uncertainty_calculator = KeypointsDistanceCalculator(image_size=config.image_size)

dataset = SPEEDplusTrainDataset(config)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
)

for batch in dataloader:
    image_tensor, image, label = batch
    keypoints_image_label = label["points_image"] # (B, N, 2)
    error = 0.1 * torch.randn_like(keypoints_image_label)
    for b in range(keypoints_image_label.shape[0]):
        error0_idx = random.sample(range(0, 10), 5)
        error[b, error0_idx, :] = 0.0
    keypoints_image_pre = keypoints_image_label + error
    uncertainty_label = uncertainty_calculator(
        keypoints_decode_pre=keypoints_image_pre,
        keypoints_decode_label=keypoints_image_label
    )
    t, q = pose_decoder.decode2pose(
        keypoints_decode_pre=keypoints_image_pre,  # (B, N, 2)
        uncertainty_pre=uncertainty_label  # (B, N)
    )
    print(f"label: {label['pos']} {label['ori']}")
    print(f"pred: {t} {q}")