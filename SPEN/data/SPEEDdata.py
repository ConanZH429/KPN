import json
import torch

import cv2 as cv
import numpy as np
import albumentations as A

from threading import Thread
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from random import shuffle
from torchvision.transforms import v2, InterpolationMode
from ..cfg import SPEEDConfig
from .augmentation import DropBlockSafe, CropAndPadSafe, CropAndPaste, AlbumentationAug, OpticalCenterRotation, TransRotation, SurfaceBrightnessAug, ClothSurfaceAug, SunFlare
from .utils import MultiEpochsDataLoader, world2image, points2box
from ..utils import SPEEDCamera


from typing import List, Dict, Tuple



def SPEED_split_dataset(config: SPEEDConfig):
    """
    Split the SPEED dataset into train and val set.

    Args:
        config (SPEEDConfig): The config object.
    
    Returns:
        None
    """
    dataset_folder = config.dataset_folder
    train_txt_path = dataset_folder / "train.txt"
    val_txt_path = dataset_folder / "val.txt"
    train_json_path = dataset_folder / "train_label.json"
    val_json_path = dataset_folder / "val_label.json"
    if train_txt_path.exists() and val_txt_path.exists():
        if train_json_path.exists() and val_json_path.exists():
            return None
    
    with open(dataset_folder / "train.json", "r") as f:
        label = json.load(f)
    total_count = len(label)
    for key in label.keys():
        label[key]["d"] = np.linalg.norm(np.array(label[key]["pos"]))
    d_list = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    d_count = [0] * len(d_list)
    count_dict_key = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35", "35-40", "40-45", "45-50"]
    count_dict = {
        "0-5": [],
        "5-10": [],
        "10-15": [],
        "15-20": [],
        "20-25": [],
        "25-30": [],
        "30-35": [],
        "35-40": [],
        "40-45": [],
        "45-50": [],
    }
    for key in label.keys():
        for i in range(len(d_list) - 1):
            if label[key]["d"] >= d_list[i] and label[key]["d"] < d_list[i+1]:
                d_count[i] += 1
                count_dict[count_dict_key[i]].append(key)
    for key in count_dict.keys():
        shuffle(count_dict[key])
    train_count_all = int(total_count * config.train_ratio)
    val_count_all = total_count - train_count_all
    train_list = count_dict["45-50"]
    train_list = train_list + count_dict["40-45"]
    val_list = []
    train_count = int(len(count_dict["35-40"]) * config.train_ratio)
    train_list = train_list + count_dict["35-40"][:train_count]
    val_list = val_list + count_dict["35-40"][train_count:]

    train_count = int(len(count_dict["30-35"]) * config.train_ratio)
    train_list = train_list + count_dict["30-35"][:train_count]
    val_list = val_list + count_dict["30-35"][train_count:]

    train_count = int(len(count_dict["25-30"]) * config.train_ratio)
    train_list = train_list + count_dict["25-30"][:train_count]
    val_list = val_list + count_dict["25-30"][train_count:]

    train_count = int(len(count_dict["20-25"]) * config.train_ratio)
    train_list = train_list + count_dict["20-25"][:train_count]
    val_list = val_list + count_dict["20-25"][train_count:]

    train_count = int(len(count_dict["15-20"]) * config.train_ratio)
    train_list = train_list + count_dict["15-20"][:train_count]
    val_list = val_list + count_dict["15-20"][train_count:]

    train_count = int(len(count_dict["10-15"]) * config.train_ratio)
    train_list = train_list + count_dict["10-15"][:train_count]
    val_list = val_list + count_dict["10-15"][train_count:]

    train_count = int(len(count_dict["5-10"]) * config.train_ratio)
    train_list = train_list + count_dict["5-10"][:train_count]
    val_list = val_list + count_dict["5-10"][train_count:]

    
    train_count = train_count_all - len(train_list)
    train_list = train_list + count_dict["0-5"][:train_count]
    val_list = val_list + count_dict["0-5"][train_count:]
    train_label = {k: label[k] for k in train_list}
    val_label = {k: label[k] for k in val_list}
    with open(train_txt_path, "w") as f:
        f.write("\n".join(train_list))
    with open(val_txt_path, "w") as f:
        f.write("\n".join(val_list))
    with open(dataset_folder / "train_label.json", "w") as f:
        json.dump(train_label, f)
    with open(dataset_folder / "val_label.json", "w") as f:
        json.dump(val_label, f)
    return None


class ImageReader(Thread):
    def __init__(self, image_name: List, image_size: Tuple[int, int], dataset_folder: Path):
        Thread.__init__(self)
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.image_name =  image_name
        self.image_dict: dict = {}
    
    def run(self):
        for image_name in tqdm(self.image_name):
            image = cv.imread(str(self.dataset_folder / "images" / "train" / image_name), cv.IMREAD_GRAYSCALE)
            if self.image_size is not None:
                image = cv.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv.INTER_LINEAR)
            self.image_dict[image_name] = image
    
    def get_result(self) -> dict:
        return self.image_dict


class SPEEDDataset(Dataset):
    """
    The base class for SPEED dataset.
    """
    def __init__(self, config: SPEEDConfig, mode: str = "train"):
        super().__init__()
        SPEED_split_dataset(config)
        self.mode = mode
        self.dataset_folder = config.dataset_folder
        self.cache = config.cache
        self.image_size = config.image_size
        self.resize_first = config.resize_first
        self.image_first_size = config.image_first_size
        self.Camera = SPEEDCamera(config.image_first_size) if self.resize_first else SPEEDCamera((1200, 1920))
        # load the label
        with open(self.dataset_folder / f"{mode}_label.json", "r") as f:
            self.label = json.load(f)
        self.image_list = list(self.label.keys())
        self.ratio = config.val_ratio if mode == "val" else config.train_ratio
        # cache the image data
        if self.cache:
            self.image_dict = {}
            self._cache_image_multithread(self.image_list)
            print(f"Load {len(self.image_dict)} {mode} images ({self.image_dict[self.image_list[0]].shape}) successfully.")
        # transform the value of label to numpy array
        for k in self.label.keys():
            self.label[k]["pos"] = np.array(self.label[k]["pos"], dtype=np.float32)
            self.label[k]["ori"] = np.array(self.label[k]["ori"], dtype=np.float32)
        # caculate the keypoints and bbox of the image
        for k in self.label.keys():
            points_cam, points_image, r_cam_min_idx, r_cam_max_idx = world2image(self.label[k]["pos"], self.label[k]["ori"], self.Camera)
            self.label[k]["points_cam"] = points_cam      # 11x4
            self.label[k]["points_image"] = points_image      # 11x3
            self.label[k]["r_cam_min_idx"] = r_cam_min_idx
            self.label[k]["r_cam_max_idx"] = r_cam_max_idx
            box, in_image_num = points2box(points_image, self.image_first_size if self.resize_first else (1200, 1920))
            self.label[k]["bbox"] = box
            self.label[k]["in_image_num"] = in_image_num
        # transform the image to tensor
        self.image2tensor = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True)
        ])
        # encoder
        if self.mode in ("train", "test"):
            self.pos_encoder_list = [
                get_pos_encoder(pos_type, **config.pos_args[pos_type])
                for pos_type in config.pos_loss_dict.keys()
            ]
            self.ori_encoder_list = [
                get_ori_encoder(ori_type, **config.ori_args[ori_type])
                for ori_type in config.ori_loss_dict.keys()
            ]
        self.len = int(len(self.image_list))
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int):
        raise NotImplementedError("Subclass of SPEEDDataset should implement __getitem__ method.")

    def _get_image(self, image_name: str) -> np.ndarray:
        """
        Get the image data from the the image dict if cache is True,
        otherwise load the image from the disk.

        Args:
            image_name (str): The image name.
        
        Returns:
            np.ndarray: The image data.
        """
        if self.cache:
            return self.image_dict[image_name]
        else:
            image = cv.imread(str(self.dataset_folder / "images" / "train" / image_name), cv.IMREAD_GRAYSCALE)
            if self.resize_first:
                image = cv.resize(image, (self.image_first_size[1], self.image_first_size[0]), interpolation=cv.INTER_LINEAR)
            return image
    
    def _get_label(self, image_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the label data from the label dict.

        Args:
            image_name (str): The image name.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The position, orientation and bounding box.
        """
        label = self.label[image_name]
        return label["pos"], label["ori"], label["bbox"], label["points_cam"], label["points_image"], label["in_image_num"], label["r_cam_min_idx"], label["r_cam_max_idx"]


    def divide_data(self, lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _cache_image_multithread(self, image_list: List[str]):
        image_divided = self.divide_data(image_list, 10)
        threads = []
        if self.resize_first:
            for sub_image_name in image_divided:
                threads.append(ImageReader(sub_image_name, self.image_first_size, self.dataset_folder))
        else:
            for sub_image_name in image_divided:
                threads.append(ImageReader(sub_image_name, None, self.dataset_folder))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for thread in threads:
            self.image_dict.update(thread.get_result())
        


class SPEEDTrainDataset(SPEEDDataset):
    """
    The dataset class for SPEED train set.
    """
    def __init__(self, config: SPEEDConfig):
        super().__init__(config, "train")
        self.crop_and_paste = CropAndPaste(p=config.CropAndPaste_p)
        self.crop_and_pad_safe = CropAndPadSafe(p=config.CropAndPadSafe_p)
        self.drop_block_safe = DropBlockSafe(p=config.DropBlockSafe_p, **config.DropBlockSafe_args)
        self.sun_flare = SunFlare(p=config.SunFlare_p)
        self.cloth_surface = ClothSurfaceAug(
            image_shape=self.image_first_size if self.resize_first else self.image_size,
            p=config.ClothSurface_p,
        )
        self.surface_brightness = SurfaceBrightnessAug(p=config.SurfaceBrightness_p)
        self.z_axis_rotation = ZAxisRotation(
            image_shape=self.image_first_size if self.resize_first else self.image_size,
            p=config.ZAxisRotation_p,
            Camera=self.Camera,
            **config.ZAxisRotation_args
        )
        self.optical_center_rotation = OpticalCenterRotation(
            image_shape=self.image_first_size if self.resize_first else self.image_size,
            p=config.OpticalCenterRotation_p,
            Camera=self.Camera,
            **config.OpticalCenterRotation_args
        )
        self.trans_rotation = TransRotation(
            image_shape=self.image_first_size if self.resize_first else self.image_size,
            p=config.TransRotation_p,
            Camera=self.Camera,
            **config.TransRotation_args
        )
        self.albumentation_aug = AlbumentationAug(p=config.AlbumentationAug_p)

    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box, points_cam, points_image, in_image_num, r_cam_min_idx, r_cam_max_idx = self._get_label(self.image_list[index])

        # data augmentation
        image = self.crop_and_paste(image, box)
        image = self.crop_and_pad_safe(image, box)
        image = self.drop_block_safe(image, box)
        image = self.cloth_surface(image, points_image, r_cam_min_idx)
        image = self.surface_brightness(image, points_image, r_cam_min_idx)
        image = self.sun_flare(image, box)
        image, pos, ori, box, points_cam, points_image = self.z_axis_rotation(image, pos, ori, box, points_cam, points_image, in_image_num)
        image, pos, ori, box, points_cam, points_image = self.optical_center_rotation(image, pos, ori, box, points_cam, points_image, in_image_num)
        image, pos, ori, box, points_cam, points_image = self.trans_rotation(image, pos, ori, box, points_cam, points_image, in_image_num, r_cam_min_idx, r_cam_max_idx)
        image = self.albumentation_aug(image, box)

        # transform the image to tensor
        image_tensor = self.image2tensor(image)
        
        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
            "points_cam": points_cam.astype(np.float32),
            "points_image": points_image.astype(np.float32),
        }
        # encode the position
        pos_encode = {}
        for pos_encoder in self.pos_encoder_list:
            pos_encode.update(pos_encoder.encode(pos))
        label["pos_encode"] = pos_encode
        # encode the orientation
        ori_encode = {}
        for ori_encoder in self.ori_encoder_list:
            ori_encode.update(ori_encoder.encode(ori))
        label["ori_encode"] = ori_encode

        return image_tensor, image, label


class SPEEDValDataset(SPEEDDataset):
    """
    The dataset class for SPEED val set.
    """
    def __init__(self, config: SPEEDConfig):
        super().__init__(config, "val")
    
    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box, points_cam, points_image, in_image_num, r_cam_min_idx, r_cam_max_idx = self._get_label(self.image_list[index])

        # transform the image to tensor
        image_tensor = self.image2tensor(image)

        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
            "points_cam": points_cam.astype(np.float32),
            "points_image": points_image.astype(np.float32),
        }

        return image_tensor, image, label


class SPEEDTestDataset(SPEEDDataset):
    """
    The dataset class for SPEED val set.
    """
    def __init__(self, config: SPEEDConfig):
        config.cache = False
        super().__init__(config, "val")
    
    def __getitem__(self, index):
        image = self._get_image(self.image_list[index])
        pos, ori, box, points_cam, points_image, in_image_num, r_cam_min_idx, r_cam_max_idx = self._get_label(self.image_list[index])

        # transform the image to tensor
        image_tensor = self.image2tensor(image)

        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
            "points_cam": points_cam.astype(np.float32),
            "points_image": points_image.astype(np.float32),
        }
        # encode the position
        pos_encode = {}
        for pos_encoder in self.pos_encoder_list:
            pos_encode.update(pos_encoder.encode(pos))
        label["pos_encode"] = pos_encode
        # encode the orientation
        ori_encode = {}
        for ori_encoder in self.ori_encoder_list:
            ori_encode.update(ori_encoder.encode(ori))
        label["ori_encode"] = ori_encode

        return image_tensor, image, label


def get_speed_dataloader(config: SPEEDConfig = SPEEDConfig()):
    data_loader = DataLoader if config.debug else MultiEpochsDataLoader
    # data_loader = DataLoader
    train_dataset = SPEEDTrainDataset(config)
    val_dataset = SPEEDValDataset(config)
    test_dataset = SPEEDTestDataset(config)
    train_loader = data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4
    )
    val_loader = data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4
    )
    return train_loader, val_loader, test_loader