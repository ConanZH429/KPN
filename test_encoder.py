from SPEN.pose import DiscreteEulerEncoder, DiscreteEulerDecoder
import numpy as np
import torch


encoder = DiscreteEulerEncoder(5, 0.1, 2, device="cpu")
decoder = DiscreteEulerDecoder(5, 0.1, 2, device="cpu")

error_sum = 0

s = 1000

for i in range(s):
    ori = np.random.rand(4)
    ori = ori / np.linalg.norm(ori)
    print(ori)

    encoded = encoder.encode(ori)

    encoded["discrete_yaw"] = torch.tensor(encoded["discrete_yaw"]).unsqueeze(0)
    encoded["discrete_pitch"] = torch.tensor(encoded["discrete_pitch"]).unsqueeze(0)
    encoded["discrete_roll"] = torch.tensor(encoded["discrete_roll"]).unsqueeze(0)

    decoded = decoder.decode_batch(encoded)
    decoded = decoded.squeeze(0).numpy()

    error = 2 * np.arccos(np.clip(np.dot(ori, decoded), -1.0, 1.0))
    print(decoded)
    print(error)
    error_sum += error

print("average error: ", error_sum / s)