import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader
from SPEN.cfg import SPEEDplusConfig
from SPEN.data import SPEEDplusTrainDataset
from SPEN.utils import SPEEDplusCamera
from scipy.spatial.transform import Rotation as R

config = SPEEDplusConfig()
config.cache = False
config.resize_first = True

config.CropAndPaste_p = 0.0
config.CropAndPadSafe_p = 0.0
config.DropBlockSafe_p = 0.0
config.ClothSurface_p = 0.0
config.SurfaceBrightness_p = 0.0
config.SunFlare_p = 0.0
config.OpticalCenterRotation_p = 0.0
config.TransRotation_p = 0.0
config.AlbumentationAug_p = 0.0
dataset = SPEEDplusTrainDataset(config)
train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)

camera = SPEEDplusCamera(config.image_size)

# plot_pose_range(dataloaders, t=100)
for batch in train_dataloader:
    image_tensor, image, label = batch
    image = image.squeeze().numpy()
    pos = label["pos"].squeeze().numpy()
    ori = label["ori"].squeeze().numpy()
    box = label["box"].squeeze().numpy()
    points_cam = label["points_cam"].squeeze().numpy()  # (N, 3)
    points_img = label["points_image"].squeeze().numpy()  # (N, 2)
    points_world = config.keypoints # (N, 3)
    points_world = points_world[:5]
    points_img = points_img[:5]
    dist = np.zeros((5, 1), dtype=np.float32)
    _, R_exp, t = cv.solvePnP(
        points_world,
        points_img,
        camera.K,
        dist,
        rvec=None,
        tvec=None,
        useExtrinsicGuess=False,
        flags=cv.SOLVEPNP_EPNP
    )
    R_pr, _ = cv.Rodrigues(R_exp)

    q = R.from_matrix(R_pr).as_quat(canonical=True, scalar_first=True)
    t = t.squeeze()
    print(f"pos: {pos}, ori: {ori}, t: {t}, q: {q}")
    break