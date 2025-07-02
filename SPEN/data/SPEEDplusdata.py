import json
import torch
import sys

import cv2 as cv
import numpy as np
import albumentations as A


from threading import Thread
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision.transforms import v2, InterpolationMode
from ..cfg import SPEEDplusConfig
from .augmentation import DropBlockSafe, CropAndPadSafe, CropAndPaste, AlbumentationAug, OpticalCenterRotation, TransRotation, SurfaceBrightnessAug, ClothSurfaceAug, SunFlare
from .utils import MultiEpochsDataLoader, world2image, points2box
from ..keypoints import get_keypoints_encoder
from ..utils import SPEEDplusCamera


from typing import List, Dict, Tuple


class ImageReader(Thread):
    def __init__(self, start: int, self_image_numpy, image_name: List, mode: str, image_size: Tuple[int, int], dataset_folder: Path):
        Thread.__init__(self)
        self.start_index = start
        self.mode = mode
        self.image_size = image_size
        self.dataset_folder = dataset_folder
        self.image_name =  image_name
        self.image_numpy = self_image_numpy
    
    def run(self):
        flag = self.mode == "val"
        image_list = []
        for (i, image_name) in tqdm(enumerate(self.image_name)):
            if flag:
                domain = image_name.split("_")[0]
                image_real_name = image_name.split("_")[1]
                image = cv.imread(str(self.dataset_folder / domain / "images" / image_real_name), cv.IMREAD_GRAYSCALE)
            else:
                image = cv.imread(str(self.dataset_folder / "synthetic" / "images" / image_name), cv.IMREAD_GRAYSCALE)
            if self.image_size is not None:
                image = cv.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv.INTER_LINEAR)
            self.image_numpy[self.start_index+i, :, :] = image
    
    def get_result(self) -> List:
        return self.image_numpy



class SPEEDplusDataset(Dataset):
    """
    The base class for SPEED dataset.
    """
    def __init__(self, config: SPEEDplusConfig, mode: str = "train"):
        super().__init__()
        self.mode = mode
        self.dataset_folder = config.dataset_folder
        self.cache = config.cache
        self.image_size = config.image_size
        self.resize_first = config.resize_first
        self.image_first_size = config.image_first_size
        self.keypoints = config.keypoints
        self.camera = SPEEDplusCamera(config.image_first_size) if self.resize_first else SPEEDplusCamera((1200, 1920))
        self.input_image_size_ratio = self.image_size[0] / (self.image_first_size[0] if self.resize_first else 1200)
        # load the labels
        self.label = {}
        if mode == "train":
            with open(self.dataset_folder / "synthetic" / "train_new.json", "r") as f:
                synthetic_train_label = json.load(f)
                self.label.update(synthetic_train_label)
            with open(self.dataset_folder / "synthetic" / "validation_new.json", "r") as f:
                synthetic_val_label = json.load(f)
                self.label.update(synthetic_val_label)
        elif mode == "val":
            with open(self.dataset_folder / "sunlamp" / "test_new.json", "r") as f:
                sumlamp_val_label = json.load(f)
                for filename, label in sumlamp_val_label.items():
                    self.label["sunlamp_" + filename] = label
            with open(self.dataset_folder / "lightbox" / "test_new.json", "r") as f:
                lightbox_val_label = json.load(f)
                for filename, label in lightbox_val_label.items():
                    self.label["lightbox_" + filename] = label
        self.image_list = list(self.label.keys())
        # cache the image data
        if self.cache:
            if self.resize_first:
                self.image_numpy = np.zeros((len(self.image_list), self.image_first_size[0], self.image_first_size[1]), dtype=np.uint8)
            else:
                self.image_numpy = np.zeros((len(self.image_list), 1200, 1920), dtype=np.uint8)
            self._cache_image_multithread(self.image_list)
            print(f"Load {self.image_numpy.shape[0]} {mode} images ({self.image_numpy[0].shape}) successfully.")
        # transform the value of label to numpy array
        for k in self.label.keys():
            self.label[k]["pos"] = np.array(self.label[k]["pos"], dtype=np.float32)
            self.label[k]["ori"] = np.array(self.label[k]["ori"], dtype=np.float32)        
        # caculate the keypoints and bbox of the image
        for k in self.label.keys():
            points_cam, points_image, points_vis, r_cam_min_idx, r_cam_max_idx = world2image(self.keypoints, self.label[k]["pos"], self.label[k]["ori"], self.camera,
                                                                                 config.image_first_size if self.resize_first else (1200, 1920))
            self.label[k]["points_cam"] = points_cam      # 11x3
            self.label[k]["points_image"] = points_image      # 11x2
            self.label[k]["points_vis"] = points_vis      # 11
            self.label[k]["r_cam_min_idx"] = r_cam_min_idx
            self.label[k]["r_cam_max_idx"] = r_cam_max_idx
            box = points2box(points_image, points_vis, self.image_first_size if self.resize_first else (1200, 1920))
            self.label[k]["bbox"] = box
        # transform the image to tensor
        self.image2tensor = v2.Compose([
            v2.ToImage(),
            v2.Resize(self.image_size, interpolation=InterpolationMode.BILINEAR),
            v2.ToDtype(torch.float32, scale=True),
        ])
        # encoder
        if self.mode in ("train", "test"):
            self.keypoints_encoder = get_keypoints_encoder(
                keypoints_type=config.keypoints_type,
                keypoints_num=config.keypoints.shape[0],
                input_image_shape=config.image_size,
                **config.keypoints_args[config.keypoints_type]
            )
        self.len = int(len(self.image_list))
    
    def __len__(self) -> int:
        return self.len
    
    def __getitem__(self, index: int):
        raise NotImplementedError("Subclass of SPEEDDataset should implement __getitem__ method.")

    def _get_image(self, index: int, image_name: str) -> np.ndarray:
        """
        Get the image data from the the image dict if cache is True,
        otherwise load the image from the disk.

        Args:
            image_name (str): The image name.
        
        Returns:
            np.ndarray: The image data.
        """
        if self.cache:
            return self.image_numpy[index]
        else:
            if self.mode == "val":
                domain = image_name.split("_")[0]
                image_real_name = image_name.split("_")[1]
                image = cv.imread(str(self.dataset_folder / domain / "images" / image_real_name), cv.IMREAD_GRAYSCALE)
            else:
                image = cv.imread(str(self.dataset_folder / "synthetic" / "images" / image_name), cv.IMREAD_GRAYSCALE)
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
        return label["pos"], label["ori"], label["bbox"], label["points_cam"], label["points_image"], label["points_vis"], label["r_cam_min_idx"], label["r_cam_max_idx"]


    def divide_data(self, lst: list, n: int):
        # 将列表lst分为n份，最后不足一份单独一组
        k, m = divmod(len(lst), n)
        return (lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

    def _cache_image_multithread(self, image_list: List[str]):
        image_divided = self.divide_data(image_list, 10)
        threads = []
        start = 0
        if self.resize_first:
            for (i, sub_image_name) in enumerate(image_divided):
                threads.append(ImageReader(start, self.image_numpy, sub_image_name, self.mode, self.image_first_size, self.dataset_folder))
                start += len(sub_image_name)
        else:
            for (i, sub_image_name) in enumerate(image_divided):
                threads.append(ImageReader(start, self.image_numpy, sub_image_name, self.mode, None, self.dataset_folder))
                start += len(sub_image_name)
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


class SPEEDplusTrainDataset(SPEEDplusDataset):
    """
    The dataset class for SPEED train set.
    """
    def __init__(self, config: SPEEDplusConfig):
        super().__init__(config, "train")
        self.crop_and_paste = CropAndPaste(p=config.CropAndPaste_p)
        self.crop_and_pad_safe = CropAndPadSafe(p=config.CropAndPadSafe_p)
        self.drop_block_safe = DropBlockSafe(p=config.DropBlockSafe_p, **config.DropBlockSafe_args)
        self.sun_flare = SunFlare(p=config.SunFlare_p)
        self.cloth_surface = ClothSurfaceAug(
            p=config.ClothSurface_p,
        )
        self.surface_brightness = SurfaceBrightnessAug(p=config.SurfaceBrightness_p)
        self.optical_center_rotation = OpticalCenterRotation(
            image_shape=self.image_first_size if self.resize_first else (1200, 1920),
            p=config.OpticalCenterRotation_p,
            camera=self.camera,
            **config.OpticalCenterRotation_args
        )
        self.albumentation_aug = AlbumentationAug(p=config.AlbumentationAug_p)

    def __getitem__(self, index):
        image = self._get_image(index, self.image_list[index])
        pos, ori, box, points_cam, points_image, points_vis, r_cam_min_idx, r_cam_max_idx = self._get_label(self.image_list[index])
        
        # data augmentation
        image = self.crop_and_paste(image, box)
        image = self.crop_and_pad_safe(image, box)
        image = self.drop_block_safe(image, box)
        image = self.cloth_surface(image, points_image, r_cam_min_idx)
        image = self.surface_brightness(image, points_image, r_cam_min_idx)
        image = self.sun_flare(image, box)
        image, pos, ori, box, points_cam, points_image, points_vis = self.optical_center_rotation(image, pos, ori, box, points_cam, points_image, points_vis)
        image = self.albumentation_aug(image, box)

        # transform the image to tensor
        image_tensor = self.image2tensor(image)
        
        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
            "points_cam": points_cam.astype(np.float32),
            "points_image": points_image.astype(np.float32) * self.input_image_size_ratio,
            "points_vis": points_vis,
        }
        # encode keypoints
        label["keypoints_encode"] = self.keypoints_encoder.encode(label["points_image"])

        return image_tensor, image, label


class SPEEDplusValDataset(SPEEDplusDataset):
    """
    The dataset class for SPEED val set.
    """
    def __init__(self, config: SPEEDplusConfig):
        super().__init__(config, "val")
    
    def __getitem__(self, index):
        image = self._get_image(index, self.image_list[index])
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

class SPEEDplusTestDataset(SPEEDplusDataset):
    """
    The dataset class for SPEED test set.
    """
    def __init__(self, config: SPEEDplusConfig):
        config.cache = False
        super().__init__(config, "test")
    
    def __getitem__(self, index):
        image = self._get_image(index, self.image_list[index])
        pos, ori, box, points_cam, points_image, in_image_num, r_cam_min_idx, r_cam_max_idx = self._get_label(self.image_list[index])

        # transform the image to tensor
        image_tensor = self.image2tensor(image)

        label = {
            "image_name": self.image_list[index],
            "pos": pos.astype(np.float32),
            "ori": ori.astype(np.float32),
            "box": box.astype(np.int32),
            "points_cam": points_cam.astype(np.float32),
            "points_image": points_image.astype(np.float32) * self.input_image_size_ratio,
        }
        # encode the position
        label["keypoints_encode"] = self.keypoints_encoder.encode(label["points_image"])

        return image_tensor, image, label

def get_speedplus_dataloader(config: SPEEDplusConfig):
    data_loader = DataLoader if config.debug else MultiEpochsDataLoader
    # data_loader = DataLoader
    train_dataset = SPEEDplusTrainDataset(config)
    val_dataset = SPEEDplusValDataset(config)
    test_dataset = SPEEDplusTestDataset(config)
    train_loader = data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4,
        drop_last=True,
    )
    val_loader = data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=True,
        pin_memory=True,
        pin_memory_device="cuda",
        prefetch_factor=4,
        drop_last=True,
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