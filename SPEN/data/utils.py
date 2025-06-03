from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils import SPEEDCamera

def points2box(points: np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Args:
        points: (N, 3)
    """
    box = np.array(
        [
            max(points[:, 0].min(), 0),                 # x_min
            max(points[:, 1].min(), 0),                 # y_min
            min(points[:, 0].max(), image_size[1]),     # x_max
            min(points[:, 1].max(), image_size[0]),     # y_max
        ],
        dtype=np.int32
    )
    in_image_num = np.sum(
        (points[:, 0] > 0) & (points[:, 0] < image_size[1]) &
        (points[:, 1] > 0) & (points[:, 1] < image_size[0])
    )
    return box, in_image_num

def cam2image(points_cam: np.ndarray, Camera: SPEEDCamera) -> np.ndarray:
    """
    Args:
        points: (N, 4)
    """
    intrinsic_mat = np.hstack((Camera.K_image, np.zeros((3, 1))))       # 3x4
    points_image = intrinsic_mat @ points_cam.T       # 4xN
    zc = points_cam.T[2]     # N
    points_image = points_image / zc     # 3xN
    return points_image.T       # Nx3

def world2image(pos: np.ndarray, ori: np.ndarray, Camera: SPEEDCamera) -> np.ndarray:
    points = np.array(
        [[-0.37,   -0.385,   0.3215],
        [-0.37,    0.385,   0.3215],
        [ 0.37,    0.385,   0.3215],
        [ 0.37,   -0.385,   0.3215],
        [-0.37,   -0.264,   0.    ],
        [-0.37,    0.304,   0.    ],
        [ 0.37,    0.304,   0.    ],
        [ 0.37,   -0.264,   0.    ],
        [-0.5427,  0.4877,  0.2535],
        [ 0.5427,  0.4877,  0.2591],
        [ 0.305,  -0.579,   0.2515],]
    )       # 11x3
    
    points_world = np.hstack((points, np.ones((points.shape[0], 1)))).T # 4x11
    # world2cam
    rotation = R.from_quat(ori, scalar_first=True)
    extrinsic_mat = np.hstack((rotation.as_matrix(), pos.reshape(3, 1)))    # 3x4
    extrinsic_mat = np.vstack((extrinsic_mat, np.array([0, 0, 0, 1])))        # 4x4
    points_cam = extrinsic_mat @ points_world       # 4x11
    r_cam = np.linalg.norm(points_cam[:3], axis=0)     # 11
    r_cam_min_idx = r_cam[:8].argmin()     # 8
    r_cam_max_idx = r_cam[:8].argmax()     # 8
    # cam2image
    points_image = cam2image(points_cam.T, Camera).T       # 3x11
    return points_cam.T, points_image.T, r_cam_min_idx, r_cam_max_idx       # 11x4, 11x3, 8x3

class MultiEpochsDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)