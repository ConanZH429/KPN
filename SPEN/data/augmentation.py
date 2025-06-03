import cv2 as cv
import numpy as np
import albumentations as A
import random
import math

from scipy.spatial.transform import Rotation as R
from typing import Union, Tuple, List, Optional
from pathlib import Path
from ..utils import SPEEDCamera
from .sunflare import sun_flare
from .utils import points2box, world2image, cam2image
# from albumentations.augmentations.geometric.functional import perspective_bboxes

def warp_box(box, M, width, height):
    # warp points
    xy = np.ones((4, 3))

    xy[:, :2] = box[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4, 2
    )  # 所有点，矩阵变换，求变换后的点
    xy = xy @ M.T  # transform

    xy = (xy[:, :2] / xy[:, 2:3]).reshape(8)
    # create new boxes
    x = xy[[0, 2, 4, 6]]
    y = xy[[1, 3, 5, 7]]
    
    xy = np.array([x.min(), y.min(), x.max(), y.max()])
    # clip boxes
    xy[[0, 2]] = xy[[0, 2]].clip(0, width-1)
    xy[[1, 3]] = xy[[1, 3]].clip(0, height-1)
    return xy.astype(np.float32)


class SurfaceBrightnessAug():
    def __init__(
            self,
            p: float = 0.5,
    ):
        self.p = p
        self.adjacent_faces_dict = {
            0: np.array([
                [0, 1, 5, 4],
                [0, 3, 7, 4],
                [0, 1, 2, 3],
            ]),
            1: np.array([
                [1, 0, 4, 5],
                [1, 5, 6, 2],
                [1, 2, 3, 0],
            ]),
            2: np.array([
                [2, 1, 5, 6],
                [2, 3, 7, 6],
                [2, 3, 0, 1],
            ]),
            3: np.array([
                [3, 2, 6, 7],
                [3, 0, 4, 7],
                [3, 0, 1, 2],
            ]),
            4: np.array([
                [4, 5, 6, 7],
                [4, 0, 1, 5],
                [4, 7, 3, 0],
            ]),
            5: np.array([
                [5, 6, 7, 4],
                [5, 1, 2, 6],
                [5, 4, 0, 1],
            ]),
            6: np.array([
                [6, 7, 4, 5],
                [6, 2, 3, 7],
                [6, 5, 1, 2],
            ]),
            7: np.array([
                [7, 4, 5, 6],
                [7, 3, 2, 6],
                [7, 4, 0, 3],
            ])
        }
        self.aug = A.RandomBrightnessContrast(
            brightness_limit=(-0.5, 0.5),
            contrast_limit=(-0.1, 0.1),
            ensure_safe_range=True,
            p=1.0
        )

    def __call__(
            self,
            image: np.ndarray,
            points_image: np.ndarray,
            r_cam_min_idx: int,
    ):
        if random.random() > self.p:
            return image

        points_image = points_image.copy().astype(np.int32)
        output = image.copy().astype(np.uint8)

        for i in range(3):
            surface_idx = self.adjacent_faces_dict[r_cam_min_idx][i]
            output = self.change_single_surface(
                image=output,
                points_image=points_image[:8, :2],
                surface_idx=surface_idx
            )

        return output

    def change_single_surface(
            self,
            image: np.ndarray,
            points_image: np.ndarray,
            surface_idx: np.ndarray
    ):
        mask = np.zeros_like(image, dtype=np.uint8)
        mask = cv.fillConvexPoly(mask, points_image[surface_idx, :], 1)
        mask = mask.astype(bool)
        output, image_copy = image.copy(), image.copy()
        image_copy = self.aug(image=image_copy)["image"]

        output[mask] = image_copy[mask]
        return output


class ClothSurfaceAug():
    def __init__(self, image_shape: Tuple[int], p: float=0.5):
        self.p = p
        # cache the cloth surface
        cloth_folder = Path(__file__).parent.resolve() / "cloth"
        self.cloth_images = []
        self.cloth_size = []
        for file in cloth_folder.iterdir():
            image = cv.imread(str(file), cv.IMREAD_GRAYSCALE)
            self.cloth_images.append(image)
        self.cloth_num = len(self.cloth_images)
        self.adjacent_faces_dict = {
            0: np.array([
                [0, 1, 5, 4],
                [0, 3, 7, 4],
            ]),
            1: np.array([
                [1, 0, 4, 5],
                [1, 5, 6, 2],
            ]),
            2: np.array([
                [2, 1, 5, 6],
                [2, 3, 7, 6],
            ]),
            3: np.array([
                [3, 2, 6, 7],
                [3, 0, 4, 7],
            ]),
            4: np.array([
                [4, 5, 6, 7],
                [4, 0, 1, 5],
                [4, 7, 3, 0],
            ]),
            5: np.array([
                [5, 6, 7, 4],
                [5, 1, 2, 6],
                [5, 4, 0, 1],
            ]),
            6: np.array([
                [6, 7, 4, 5],
                [6, 2, 3, 7],
                [6, 5, 1, 2],
            ]),
            7: np.array([
                [7, 4, 5, 6],
                [7, 3, 2, 6],
                [7, 4, 0, 3],
            ])
        }
    
    def __call__(
            self,
            image: np.ndarray,
            points_image: np.ndarray,
            r_cam_min_idx: int,
    ):
        if random.random() > self.p:
            return image

        points_image = points_image.copy().astype(np.int32)

        surface_picked = random.randint(0, len(self.adjacent_faces_dict[r_cam_min_idx]) - 1)
        surface_idx = self.adjacent_faces_dict[r_cam_min_idx][surface_picked]
        surface_points = points_image[surface_idx, :2]

        cloth_picked = random.randint(0, self.cloth_num - 1)
        cloth_image = self.cloth_images[cloth_picked]
        cloth_h, cloth_w = cloth_image.shape[:2]
        warp_matrix = cv.getPerspectiveTransform(
            np.array([[0, 0], [cloth_w, 0], [cloth_w, cloth_h], [0, cloth_h]]).astype(np.float32),
            np.array(surface_points).astype(np.float32)
        )

        cloth_image = cv.warpPerspective(cloth_image, warp_matrix, (image.shape[1], image.shape[0]), flags=cv.INTER_LINEAR)

        mask = np.zeros_like(image, dtype=np.uint8)
        mask = cv.fillConvexPoly(mask, surface_points, 1)
        mask = mask.astype(bool)

        image[mask] = 0.7 * image[mask] + 0.3 * cloth_image[mask]
        return image
    

class TransRotation():
    def __init__(
            self,
            image_shape: Tuple[int],
            max_angle: float,
            max_trans_xy: float,
            max_trans_z: float,
            Camera,
            max_t: int,
            p: float = 1.0
    ):
        self.max_angle = max_angle
        self.max_trans_xy = max_trans_xy
        self.max_trans_z = max_trans_z
        self.Camera = Camera
        self.max_t = max_t
        self.p = p
        self.key_points = np.array(
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
        self.image_shape = image_shape
    
    def __call__(
            self,
            image: np.ndarray,
            pos: np.ndarray,
            ori: np.ndarray,
            box: np.ndarray,
            points_cam: np.ndarray,
            points_image: np.ndarray,
            in_image_num: int,
            r_cam_min_idx: int,
            r_cam_max_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply translation and rotation to an image and its corresponding position, orientation and bounding box.

        Args:
            image (np.ndarray): The image.
            pos (np.ndarray): The position.
            ori (np.ndarray): The orientation.
            box (np.ndarray): The bounding box.
            points_cam (np.ndarray): The points in camera coordinates, shape: (11, 4).
            points_image (np.ndarray): The points in image coordinates, shape: (11, 3).
            in_image_num (int): The number of points in the image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The augmented image, position, orientation, bounding box, points in camera coordinates and points in image coordinates.
        """
        if random.random() > self.p:
            return image, pos, ori, box, points_cam, points_image
        
        h, w = self.image_shape

        t = 0
        while True:
            # translation
            delta = np.array([
                random.uniform(-self.max_trans_xy, self.max_trans_xy),
                random.uniform(-self.max_trans_xy, self.max_trans_xy),
                random.uniform(-self.max_trans_z, self.max_trans_z)
            ]) + 1
            
            pos_warped = pos * delta
            pos_warped[2] = np.clip(pos_warped[2], 5, 35)

            x_lim = self.Camera.K_image[0, 2] * pos_warped[2] / self.Camera.K_image[0, 0]
            y_lim = self.Camera.K_image[1, 2] * pos_warped[2] / self.Camera.K_image[1, 1]
            pos_warped[0] = np.clip(pos_warped[0], -x_lim, x_lim)
            pos_warped[1] = np.clip(pos_warped[1], -y_lim, y_lim)

            # rotation
            delta_yaw = random.uniform(-self.max_angle, self.max_angle)     # degree
            delta_pitch = random.uniform(-self.max_angle, self.max_angle)   # degree
            delta_roll = random.uniform(-self.max_angle, self.max_angle)    # degree
            delta_euler = np.array([delta_yaw, delta_pitch, delta_roll])

            rotation = R.from_quat(ori, scalar_first=True)
            euler = rotation.as_euler("YXZ", degrees=True)
            euler_warped = euler + delta_euler
            rotation_warped = R.from_euler("YXZ", euler_warped, degrees=True)
            rotation_matrix_warped = rotation_warped.as_matrix()

            extrinsic_matrix = np.hstack((rotation_matrix_warped, pos_warped.reshape(3, 1)))    # 3x4
            extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
            points_world = np.hstack((self.key_points, np.ones((self.key_points.shape[0], 1)))).T # 4x11
            points_cam_warped = extrinsic_matrix @ points_world     # 4x11
            r_cam = np.linalg.norm(points_cam_warped[:3], axis=0)     # 11
            r_cam_min_idx_warped = r_cam[:8].argmin()     # 11

            points_image_warped = cam2image(points_cam_warped.T, self.Camera).T       # 3x11
            box_warped, in_image_num_warped = points2box(points_image_warped.T, (h, w))

            if in_image_num_warped == in_image_num and r_cam_min_idx_warped == r_cam_min_idx:
                break
            else:
                t += 1
                if t > self.max_t:
                    return image, pos, ori, box, points_cam, points_image
        
        points_mask = np.ones(11, dtype=bool)
        points_mask[r_cam_max_idx] = False
        points_mask[-3:] = False

        warp_matrix, mask = cv.findHomography(
            points_image[points_mask, :2].astype(np.float32),
            points_image_warped.T[points_mask, :2].astype(np.float32)
        )
        if np.any(mask == 0):
            return image, pos, ori, box, points_cam, points_image

        image_warped = cv.warpPerspective(image, warp_matrix, (w, h), flags=cv.INTER_LINEAR)

        return image_warped, pos_warped.astype(np.float32), ori.astype(np.float32), box_warped.astype(np.int32), points_cam_warped.T.astype(np.int32), points_image_warped.T.astype(np.int32)


class OpticalCenterRotation():

    def __init__(
            self,
            image_shape: Tuple[int],
            max_angle: float,
            Camera,
            max_t: int = 5,
            p: float = 1.0
    ):
        self.max_angle = max_angle
        self.max_t = max_t
        self.Camera = Camera
        self.p = p
        self.image_shape = image_shape

    def __call__(
            self,
            image: np.ndarray,
            pos: np.ndarray,
            ori: np.ndarray,
            box: np.ndarray,
            points_cam: np.ndarray,
            points_image: np.ndarray,
            in_image_num: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate around a line passing through the camera's optical center.

        Args:
            image (np.ndarray): The image.
            pos (np.ndarray): The position.
            ori (np.ndarray): The orientation.
            box (np.ndarray): The bounding box.
            points_cam (np.ndarray): The points in camera coordinates, shape: (11, 4).
            points_image (np.ndarray): The points in image coordinates, shape: (11, 3).
            in_image_num (int): The number of points in the image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: The augmented image, position, orientation, bounding box, points in camera coordinates, and points in image coordinates.
        """

        if random.random() > self.p:
            return image, pos, ori, box, points_cam, points_image
        
        h, w = self.image_shape

        t = 0
        while True:
            rot_angle = random.uniform(-self.max_angle, self.max_angle)  # degree
            x, y, z = pos
            r = math.sqrt(x**2 + y**2 + z**2)
            theta = np.rad2deg(math.acos(z / r))
            phi = np.rad2deg(math.atan2(y, x))
            theta = np.deg2rad(random.uniform(0, min(theta+10, 20)))        # radians
            phi = np.deg2rad(random.uniform(phi-10, phi+10))    # radians
            rotvec = np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)])

            rotation = R.from_rotvec(rot_angle * rotvec, degrees=True)
            rotation_matrix = rotation.as_matrix()

            extrinsic_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
            extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
            points_cam_warped = extrinsic_matrix @ points_cam.T     # 4x11

            points_image_warped = cam2image(points_cam_warped.T, self.Camera).T       # 3x11
            box_warped, in_image_num_warped = points2box(points_image_warped.T, (h, w))

            if in_image_num_warped >= in_image_num - 1:
                break
            else:
                t += 1
                if t > self.max_t:
                    return image, pos, ori, box, points_cam, points_image
                continue
        
        warp_matrix = self.Camera.K_image @ rotation_matrix @ self.Camera.K_image_inv
        image_warped = cv.warpPerspective(image, warp_matrix, (w, h), flags=cv.INTER_LINEAR)

        pos_warped = rotation_matrix @ pos
        ori_warped = rotation * R.from_quat(ori, scalar_first=True)
        ori_warped = ori_warped.as_quat(canonical=True, scalar_first=True)

        return image_warped, pos_warped.astype(np.float32), ori_warped.astype(np.float32), box_warped.astype(np.int32), points_cam_warped.T.astype(np.int32), points_image_warped.T.astype(np.int32)


class ZAxisRotation():
    """
    Rotate the image around the z-axis.
    """
    def __init__(
            self,
            image_shape: Tuple[int],
            max_angle: float,
            Camera,
            max_t: int = 5,
            p: float = 1.0
        ):
        """
        Initialize the ZAxisRotation class.

        Args:
            max_angle (float): The maximum rotation angle. Unit: degree.
            max_t (int): The maximum number of trials.
            Camera: The camera.
            p (float): The probability of applying rotation.
        """
        self.max_angle = max_angle
        self.max_t = max_t
        self.Camera = Camera
        self.p = p
        self.image_shape = image_shape
    
    def __call__(
            self,
            image: np.ndarray,
            pos: np.ndarray,
            ori: np.ndarray,
            box: np.ndarray,
            points_cam: np.ndarray,
            points_image: np.ndarray,
            in_image_num: int
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply rotation around the z-axis to an image and its corresponding position, orientation, bounding box and key points.
        Args:
            image (np.ndarray): The image.
            pos (np.ndarray): The position. Shape: (3,).
            ori (np.ndarray): The orientation. Shape: (4,).
            box (np.ndarray): The bounding box. Shape: (4,).
            key_points (np.ndarray): The key points. Shape: (11, 3).
        """
        if random.random() > self.p:
            return image, pos, ori, box, points_cam, points_image

        h, w = self.image_shape

        t = 0
        while True:
            angle = random.uniform(-self.max_angle, self.max_angle)

            rotation = R.from_euler("YXZ", [0, 0, angle], degrees=True)
            rotation_matrix = rotation.as_matrix()

            extrinsic_matrix = np.hstack((rotation_matrix, np.zeros((3, 1))))
            extrinsic_matrix = np.vstack((extrinsic_matrix, np.array([0, 0, 0, 1])))
            points_cam_warped = extrinsic_matrix @ points_cam.T     # 4x11

            points_image_warped = cam2image(points_cam_warped.T, self.Camera).T       # 3x11
            box_warped, in_image_num_warped = points2box(points_image_warped.T, (h, w))
    
            if in_image_num_warped >= in_image_num - 1:
                break
            else:
                t += 1
                if t > self.max_t:
                    return image, pos, ori, box, points_cam, points_image

        warp_matrix = self.Camera.K_image @ rotation_matrix @ self.Camera.K_image_inv
        image_warped = cv.warpPerspective(image, warp_matrix, (w, h), flags=cv.INTER_LINEAR)

        pos_warped = rotation_matrix @ pos
        ori_warped = rotation * R.from_quat(ori, scalar_first=True)
        ori_warped = ori_warped.as_quat(canonical=True, scalar_first=True)

        return image_warped, pos_warped.astype(np.float32), ori_warped.astype(np.float32), box_warped.astype(np.int32), points_cam_warped.astype(np.int32).T, points_image_warped.astype(np.int32).T


class SunFlare():
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            box_w, box_h = box[2] - box[0], box[3] - box[1]
            src_radius = random.randint(int(min(box_w, box_h) * 0.6), int(max(box_w, box_h) * 1.2)) // 2
            image = sun_flare(
                image=image,
                flare_center=(
                    random.randint(box[0], box[2]),
                    random.randint(box[1], box[3]),
                ),
                src_radius=src_radius,
                src_color=random.randint(240, 255),
                angle_range=(0, 1),
                num_circles=random.randint(5, 10),
            )
        return image


class AlbumentationAug():
    """
    Data augmentation using albumentations.
    """
    def __init__(self, p: float, sunflare_p: float = 0.5):
        """
        Initialize the AlbumentationAug class.

        Args:
            p (float): The probability of applying data augmentation.
        """
        self.p = p
        self.sunflare_p = sunflare_p

        self.aug = A.OneOf([
            A.OneOf([
                A.MedianBlur(),
                A.MotionBlur(),
                A.GaussianBlur(),
                A.GlassBlur()
            ]),
            A.ColorJitter(),
            A.GaussNoise()
        ], p=p)
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to the image.

        Args:
            image (np.ndarray): The image.

        Returns:
            np.ndarray: The augmented image.
        """
        image = self.aug(image=image)["image"]
        
        return image


class DropBlockSafe():
    """
    Drop blocks on the image.
    """
    def __init__(self, drop_num: int, p: float):
        """
        Initialize the DropBlockSafe class.

        Args:
            drop_num (int): The maximum number of drop blocks.
            p (float): The probability of applying drop blocks
        """
        self.drop_num = drop_num
        self.p = p
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Drop blocks on the image.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.

        Returns:
            np.ndarray: The image with drop blocks.
        """
        if random.random() > self.p:
            return image
        
        # Get the image size
        h, w = image.shape[:2]
        # Get the bounding box
        x_min, y_min, x_max, y_max = box
        # The number of drop blocks
        drop_num = random.randomint(1, self.drop_num + 1)

        back_ground = random.randomint(0, 256, (h, w), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Drop blocks
        for _ in range(drop_num):
            drop_x_min = random.randomint(0, w-1)
            drop_y_min = random.randomint(0, h-1)
            drop_x_max = random.randomint(drop_x_min, w)
            drop_y_max = random.randomint(drop_y_min, h)
            mask[drop_y_min:drop_y_max, drop_x_min:drop_x_max] = 1
        
        # Drop the blocks around the bounding box
        mask[y_min:y_max, x_min:x_max] = 0
        image = image * (1 - mask) + back_ground * mask

        return image


class CropAndPadSafe():
    """
    Crop and pad the image to original size.
    """
    def __init__(self, p: float):
        self.p = p
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Crop and pad the image to original size.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.
        
        Returns:
            np.ndarray: The cropped and padded image.
        """
        if random.random() > self.p:
            return image
        
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = box
        x_min -= 10
        y_min -= 10
        x_max += 10
        y_max += 10

        # Random crop around the bounding box
        top = random.randomint(0, y_min) if y_min > 0 else 0
        bottom = random.randomint(y_max, h) if y_max < h else h
        left = random.randomint(0, x_min) if x_min > 0 else 0
        right = random.randomint(x_max, w) if x_max < w else w

        # Crop the image
        cropped_image = image[top:bottom, left:right]
        # Pad the image
        padded_image = cv.copyMakeBorder(cropped_image, top, h-bottom, left, w-right, cv.BORDER_REPLICATE)

        return padded_image


class CropAndPaste():
    """
    Crop and paste the image.
    """
    def __init__(self, p: float):
        """
        Initialize the CropAndPaste class.

        Args:
            p (float): The probability of applying crop and paste.
        """
        self.p = p
        # cache background images
        background_folder = Path(__file__).parent.resolve() / "background"
        self.background_images = []
        self.background_size = []
        for file in background_folder.iterdir():
            self.background_images.append(cv.imread(str(file), cv.IMREAD_GRAYSCALE))
            self.background_size.append(self.background_images[-1].shape[:2])
        self.background_num = len(self.background_images)
    
    def __call__(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        Crop and paste the image.

        Args:
            image (np.ndarray): The image.
            box (np.ndarray): The bounding box.
        
        Returns:
            np.ndarray: The cropped and pasted image.
        """
        if random.random() > self.p:
            return image
        
        h, w = image.shape[:2]
        x_min, y_min, x_max, y_max = box
        target_box = image[y_min:y_max, x_min:x_max]

        # Random choose a background image
        idx = random.randint(0, self.background_num)
        background = self.background_images[idx]

        # Random crop the background image
        # Minimum width and height is 1/2 of the image
        # Maximum width and height is shape of the background image
        min_h, min_w = h // 2, w // 2
        max_h, max_w = self.background_size[idx]
        crop_h = random.randomint(min_h, max_h + 1)
        crop_w = random.randomint(min_w, max_w + 1)
        h_start = random.randomint(0, max_h - crop_h + 1)
        w_start = random.randomint(0, max_w - crop_w + 1)
        background = background[h_start:h_start + crop_h, w_start:w_start + crop_w]

        # Resize the background crop to the shape of the image
        background = cv.resize(background, (w, h))

        # Paste the target box to the background
        background[y_min:y_max, x_min:x_max] = target_box

        return background