import torch
import numpy as np
from SPEN.module.utils import KeypointsDistanceCalculator

keypoints_distance_calculator = KeypointsDistanceCalculator(image_size=(480, 768))

keypoints_pre = torch.tensor([[
    [100, 150],
    [200, 250],
    [300, 350]
]], dtype=torch.float32)

keypoints_label = np.array([[
    [103, 154],
    [210, 240],
    [290, 360]
]], dtype=np.float32)

distance = keypoints_distance_calculator(keypoints_pre, keypoints_label)
print("Keypoints Distance:", distance.numpy())