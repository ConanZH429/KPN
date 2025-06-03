from SPEN.data import SPEEDTrainDataset
from SPEN.vis import display_image
from SPEN.cfg import SPEEDConfig, SPEEDplusConfig
from SPEN.utils import SPEEDCamera
from scipy.io import loadmat
from SPEN.data import SPEEDTrainDataset, SPEEDplusTrainDataset
from torch.utils.data import DataLoader
from SPEN.vis import plot_pose_range
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
config = SPEEDConfig()
config.cache = False
config.resize_first = True
config.ori_type = "Euler"
config.ori_loss_dict = {
    "Euler": "L1"
}
config.ClothSurface_p = 1.0
config.SurfaceBrightness_p = 1.0
config.SunFlare_p = 1.0
dataset = SPEEDTrainDataset(config)
train_dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
)
dataloaders.append(train_dataloader)


# plot_pose_range(dataloaders, t=100)
for batch in train_dataloader:
    image_tensor, image, label = batch
    image = image.squeeze().numpy()
    pos = label["pos"].squeeze().numpy()
    ori = label["ori"].squeeze().numpy()
    box = label["box"].squeeze().numpy()
    print(image.shape)
    display_image(image=image,
                    pos=pos,
                    ori=ori,
                    box=box,
                    points=label["points_image"].squeeze().numpy(),
                    display_points=True,
                    display_box=True)