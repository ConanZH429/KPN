from typing import Tuple

import cv2 as cv
import numpy as np
import random

import json
from typing import Union, Optional


def read_speed_data(image_name: str, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Read the image and its corresponding position, orientation and bounding box
    from the speed dataset.

    Args:
        image_name (str): The name of the image.
        config: The configuration of the dataset.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The image, position, orientation and bounding box.
    """
    label_path = config.dataset_folder / "train_label.json"
    label = json.load(open(label_path, "r"))
    if not image_name:
        image_name = random.choice(list(label.keys()))
    image_path = config.dataset_folder / "images" / "train" / image_name
    image = cv.imread(str(image_path))
    pos = np.array(label[image_name]["pos"], dtype=np.float32)
    ori = np.array(label[image_name]["ori"], dtype=np.float32)
    box = np.array(label[image_name]["bbox"], dtype=np.int32)
    return image, pos, ori, box


def read_speedplus_data(image_name: Union[str, None], image_type: str, config) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_label_path = config.dataset_folder / "synthetic" / "train_new.json"
    val_label_path = config.dataset_folder / "synthetic" / "validation_new.json"
    sumlamp_label_path = config.dataset_folder / "sunlamp" / "test_new.json"
    lightbox_label_path = config.dataset_folder / "lightbox" / "test_new.json"

    train_label = json.load(open(train_label_path, "r"))
    val_label = json.load(open(val_label_path, "r"))
    train_label.update(val_label)
    sumlamp_label = json.load(open(sumlamp_label_path, "r"))
    lightbox_label = json.load(open(lightbox_label_path, "r"))

    if not image_name:
        image_name = random.choice(list(train_label.keys()))
        label = train_label
        image_path = config.dataset_folder / "synthetic" / "images" / image_name
    else:
        if image_type is None:
            raise ValueError("image_type cannot be None!")
        if image_type == "synthetic":
            label = train_label
            image_path = config.dataset_folder / "synthetic" / "images" / image_name
        elif image_type == "sunlamp":
            label = sumlamp_label
            image_path = config.dataset_folder / "sunlamp" / "images" / image_name
        elif image_type == "lightbox":
            label = lightbox_label
            image_path = config.dataset_folder / "lightbox" / "images" / image_name

    image = cv.imread(str(image_path))
    pos = np.array(label[image_name]["pos"], dtype=np.float32)
    ori = np.array(label[image_name]["ori"], dtype=np.float32)
    box = np.array(label[image_name]["bbox"], dtype=np.int32)

    return image, pos, ori, box