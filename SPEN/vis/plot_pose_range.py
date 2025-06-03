import matplotlib.pyplot as plt
import numpy as np
from typing import List
from tqdm import tqdm

def get_pose_range(dataloader, t: int = 1):
    x_range_dict = {}
    y_range_dict = {}
    z_range_dict = {}
    x_num_dict = {}
    y_num_dict = {}
    z_num_dict = {}
    for _ in range(t):
        for batch in tqdm(dataloader):
            image_tensor, image, label = batch
            pos = label["pos"].squeeze().numpy() * 10
            euler = np.rad2deg(label["ori_encode"]["euler"].squeeze().numpy())
            x, y, z = pos.astype(np.int32)
            x = int(x)
            y = int(y)
            z = int(z)
            x_range_dict[x] = x_range_dict.get(x, set())
            y_range_dict[y] = y_range_dict.get(y, set())
            z_range_dict[z] = z_range_dict.get(z, set())

            yaw, pitch, roll = euler.astype(np.int32)
            yaw = (int(yaw) + 180) // 2
            pitch = (int(pitch) + 90) // 2
            roll = (int(roll) + 180) // 2
            encode_euler = yaw*1000000 + pitch*10000 + roll
            x_range_dict[x].add(encode_euler)
            y_range_dict[y].add(encode_euler)
            z_range_dict[z].add(encode_euler)
        x_keys = sorted(x_range_dict.keys())
        y_keys = sorted(y_range_dict.keys())
        z_keys = sorted(z_range_dict.keys())
        max_diff_x = -1
        max_diff_y = -1
        max_diff_z = -1
        for x_key in x_keys:
            x_num = len(x_range_dict[x_key])
            max_diff_x = max(max_diff_x, abs(x_num_dict.get(x_key, 0) - x_num))
            x_num_dict[x_key] = x_num
        for y_key in y_keys:
            y_num = len(y_range_dict[y_key])
            max_diff_y = max(max_diff_y, abs(y_num_dict.get(y_key, 0) - y_num))
            y_num_dict[y_key] = y_num
        for z_key in z_keys:
            z_num = len(z_range_dict[z_key])
            max_diff_z = max(max_diff_z, abs(z_num_dict.get(z_key, 0) - z_num))
            z_num_dict[z_key] = z_num
        if max_diff_x < 100 and max_diff_y < 100 and max_diff_z < 100:
            break
        else:
            print(f"{_} max x diff: {max_diff_x}  max y diff: {max_diff_y}  max z diff: {max_diff_z}")
    return x_num_dict, y_num_dict, z_num_dict


def plot_pose_range(dataloaders: List, t: int = 1):
    x_pose_ratio_list = []
    y_pose_ratio_list = []
    z_pose_ratio_list = []
    for dataloader in dataloaders:
        x_num_dict, y_num_dict, z_num_dict = get_pose_range(dataloader, t=t)
        x_keys = sorted(x_num_dict.keys())
        y_keys = sorted(y_num_dict.keys())
        z_keys = sorted(z_num_dict.keys())
        x_values = [x_num_dict[x_key] for x_key in x_keys]
        y_values = [y_num_dict[y_key] for y_key in y_keys]
        z_values = [z_num_dict[z_key] for z_key in z_keys]
        x_pose_ratio_list.append((x_keys, x_values))
        y_pose_ratio_list.append((y_keys, y_values))
        z_pose_ratio_list.append((z_keys, z_values))
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    for i, (x_keys, x_values) in enumerate(x_pose_ratio_list):
        axs[0].plot(np.array(x_keys) / 10, x_values, label=f"Dataset {i+1}")
    axs[0].set_title("X Pose Range")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Pose Ratio")
    axs[0].legend()
    axs[0].grid()
    for i, (y_keys, y_values) in enumerate(y_pose_ratio_list):
        axs[1].plot(np.array(y_keys) / 10, y_values, label=f"Dataset {i+1}")
    axs[1].set_title("Y Pose Range")
    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Pose Ratio")
    axs[1].legend()
    axs[1].grid()
    for i, (z_keys, z_values) in enumerate(z_pose_ratio_list):
        axs[2].plot(np.array(z_keys) / 10, z_values, label=f"Dataset {i+1}")
    axs[2].set_title("Z Pose Range")
    axs[2].set_xlabel("Z")
    axs[2].set_ylabel("Pose Ratio")
    axs[2].legend()
    axs[2].grid()
    plt.tight_layout()
    plt.show()