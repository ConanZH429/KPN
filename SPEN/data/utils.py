from torch.utils.data import DataLoader
import numpy as np
from scipy.spatial.transform import Rotation as R
from ..utils import Camera
from typing import Tuple

def points2box(points_image: np.ndarray, points_vis:np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Args:
        points_image: (N, 2)
    """
    box = np.array(
        [
            points_image[points_vis, 0].min(),                 # x_min
            points_image[points_vis, 1].min(),                 # y_min
            points_image[points_vis, 0].max(),     # x_max
            points_image[points_vis, 1].max(),     # y_max
        ],
        dtype=np.int32
    )
    return box


def world2camera(points_world: np.ndarray, pos: np.ndarray, ori: np.ndarray) -> np.ndarray:
    """
    Args:
        points_world: (N, 3)
        pos: (3,)
        ori: (4,) quaternion, scalar first
    """
    rotation = R.from_quat(ori, scalar_first=True)
    points_camera = rotation.as_matrix() @ points_world.T + pos.reshape(3, 1)  # 3xN
    points_camera = points_camera.T  # Nx3
    return points_camera  # Nx3


def camera2image(points_cam: np.ndarray, camera: Camera, image_size: Tuple[int]) -> np.ndarray:
    """
    Args:
        points_cam: (N, 3)
        camera: Camera
    """
    intrinsic_mat = np.hstack((camera.K, np.zeros((3, 1))))  # 3x4
    points_cam = np.hstack((points_cam, np.ones((points_cam.shape[0], 1))))  # Nx4
    points_image = intrinsic_mat @ points_cam.T  # 3xN
    zc = points_cam[:, 2]  # N
    points_image = points_image / zc  # 3xN
    points_image = points_image[:2].T  # Nx2
    points_vis = (points_image[:, 0] >= 0) & (points_image[:, 0] < image_size[1]) & (points_image[:, 1] >= 0) & (points_image[:, 1] < image_size[0])
    return points_image, points_vis  # Nx2, N


def world2image(points_world: np.ndarray, pos: np.ndarray, ori: np.ndarray, camera: Camera, image_size: Tuple[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Args:
        points_world: (N, 3)
        pos: (3,)
        ori: (4,) quaternion, scalar first
        camera: Camera
    """
    # world2camera
    points_camera = world2camera(points_world, pos, ori)    # Nx3
    r_camera = np.linalg.norm(points_camera, axis=1)        # N
    r_camera_min_idx = r_camera[:8].argmin()  # 8
    r_camera_max_idx = r_camera[:8].argmax()  # 8
    # camera2image
    points_image, points_vis = camera2image(points_camera, camera, image_size)  # Nx2
    return points_camera, points_image, points_vis, r_camera_min_idx, r_camera_max_idx  # Nx3, Nx2


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