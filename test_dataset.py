from SPEN.vis import display_image
from SPEN.cfg import SPEEDplusConfig
from SPEN.data import SPEEDplusTrainDataset
from torch.utils.data import DataLoader
from SPEN.vis import plot_pose_range
from rich import print
# config = SPEEDConfig()
# config.image_first_size = (1200, 1920)
# config.image_size = (1200, 1920)
# config.cache = False
# config.ZAxisRotation_p = 0.0
# config.CropAndPadSafe_p = 0.0
# config.AlbumentationAug_p = 0.0
# config.CropAndPaste_p = 0.0
# config.DropBlockSafe_p = 0.0
# dataset = SPEEDTrainDataset(config)
# image_tensor, image, label = dataset[1022]

# camera = SPEEDCamera(config.image_size)

# points_path = "./result_file/tangoPoints.mat"
# points = loadmat(points_path)["tango3Dpoints"].T

# # 68
# # 9592
# display_image("img009592.jpg",
#               points=points,
#               display_label_axis=False,
#               display_point=True,
#               save_path="./result_file/test2.png",)

# display_image("img000001.jpg",
#               dataset_type="SPEED+",
#               image_type="synthetic",
#               display_box=True)

dataloaders = []

# config = SPEEDConfig()
# config.cache = False
# config.resize_first = True
# config.ori_type = "Euler"
# config.ori_loss_dict = {
#     "Euler": "L1"
# }
# config.ZAxisRotation_p = 0.0
# config.OpticalCenterRotation_p = 0.0
# config.TransRotation_p = 0.0
# dataset = SPEEDTrainDataset(config)
# train_dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
# )
# dataloaders.append(train_dataloader)

# config = SPEEDConfig()
# config.cache = False
# config.resize_first = True
# config.ori_type = "Euler"
# config.ori_loss_dict = {
#     "Euler": "L1"
# }
# config.ZAxisRotation_p = 1.0
# config.OpticalCenterRotation_p = 0.0
# config.TransRotation_p = 0.0
# dataset = SPEEDTrainDataset(config)
# train_dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
# )
# dataloaders.append(train_dataloader)

# config = SPEEDConfig()
# config.cache = False
# config.resize_first = True
# config.ori_type = "Euler"
# config.ori_loss_dict = {
#     "Euler": "L1"
# }
# config.ZAxisRotation_p = 0.0
# config.OpticalCenterRotation_p = 1.0
# config.TransRotation_p = 0.0
# dataset = SPEEDTrainDataset(config)
# train_dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
# )
# dataloaders.append(train_dataloader)

# config = SPEEDConfig()
# config.cache = False
# config.resize_first = True
# config.ori_type = "Euler"
# config.ori_loss_dict = {
#     "Euler": "L1"
# }
# config.ZAxisRotation_p = 0.0
# config.OpticalCenterRotation_p = 0.0
# config.TransRotation_p = 1.0
# dataset = SPEEDTrainDataset(config)
# train_dataloader = DataLoader(
#     dataset,
#     batch_size=1,
#     shuffle=True,
# )
# dataloaders.append(train_dataloader)

# Surface Brightness
config = SPEEDplusConfig()
config.cache = False
config.resize_first = True

config.SunFlare_p = 0.0
config.SurfaceSunFlare_p = 1.0

dataset = SPEEDplusTrainDataset(config)
train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)
dataloaders.append(train_dataloader)


# plot_pose_range(dataloaders, t=100)
ratio = (config.image_first_size[0] if config.resize_first else 1200) / config.image_size[0]
for batch in train_dataloader:
    image_tensor, image, label = batch
    image = image.squeeze().numpy()
    image_name = label["image_name"][0]
    pos = label["pos"].squeeze().numpy()
    ori = label["ori"].squeeze().numpy()
    box = label["box"].squeeze().numpy()
    points_cam = label["points_cam"]
    points_image = label["points_image"] * ratio  # (N, 2)
    points_image = points_image.squeeze().numpy()
    print(f"{image_name} image shape: {image.shape}  box type: {box.dtype}  points type: {points_cam.dtype}/{points_image.dtype}")
    display_image(image=image,
                    pos=pos,
                    ori=ori,
                    box=box,
                    points=points_image,
                    display_points=True,
                    display_box=True)