import torch
from torch import Tensor
import numpy as np

class CartEncoder:
    def __init__(self):
        pass

    def encode(self, pos: np.ndarray):
        return {
            "cart": pos
        }


class CartDecoder:
    def __init__(self):
        pass

    def decode_batch(self, pos_pre_dict: dict[str, Tensor]) -> Tensor:
        return pos_pre_dict["cart"]