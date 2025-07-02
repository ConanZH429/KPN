import torch
import cv2 as cv
import numpy as np
from torch import Tensor
from scipy.spatial.transform import Rotation as R
from ..utils import SPEEDplusCamera


class KeypointsDistanceCalculator:
    def __init__(self, image_size: tuple[int, int]):
        self.image_size = image_size
        self.diagonal = (image_size[0] ** 2 + image_size[1] ** 2) ** 0.5
    
    def __call__(self, keypoints_decode_pre: Tensor, keypoints_decode_label: Tensor):
        """
        Calculate the distance between predicted and labeled keypoints.

        Args:
            keypoints_decode_pre (Tensor): Predicted keypoints, shape (B, N, 2).
            keypoints_decode_label (Tensor): Labeled keypoints, shape (B, N, 2).
        """
        d = torch.norm(keypoints_decode_pre - keypoints_decode_label, p=2, dim=-1)  # (B, N)
        d = d / self.diagonal  # Normalize by the diagonal of the image
        return d


class PoseDecoder:
    def __init__(
            self,
            image_size: tuple[int, int],
            points_world: np.ndarray,   # (N, 3) points in world coordinates
            device: str = "cpu",
        ):
        self.dist = np.zeros((5, 1), dtype=np.float32)
        self.camera = SPEEDplusCamera(image_size)
        self.points_world = points_world
        self.device = device

    def decode2pose(
            self,
            keypoints_decode_pre: Tensor, # (B, N, 2)
            uncertainty_pre: Tensor, # (B, N)
    ):
        t_list = []
        q_list = []
        sorted_idx = torch.argsort(uncertainty_pre, dim=1).cpu().numpy()
        keypoints_decode = keypoints_decode_pre.cpu().numpy()
        for b in range(keypoints_decode.shape[0]):
            points_img = keypoints_decode[b][sorted_idx[b][:5]].astype(np.float32)  # (5, 2)
            points_world = self.points_world[sorted_idx[b][:5]].astype(np.float32)  # (5, 3)
            _, R_exp, t = cv.solvePnP(
                points_world,
                points_img,
                self.camera.K,
                self.dist,
                rvec=None,
                tvec=None,
                useExtrinsicGuess=False,
                flags=cv.SOLVEPNP_EPNP
            )
            R_pr, _ = cv.Rodrigues(R_exp)
            t = t.squeeze()
            q = R.from_matrix(R_pr).as_quat(canonical=True, scalar_first=True)
            t_list.append(torch.tensor(t, device=self.device))
            q_list.append(torch.tensor(q, device=self.device))
        t = torch.stack(t_list, dim=0)
        q = torch.stack(q_list, dim=0)
        return t, q
