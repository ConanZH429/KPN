from pathlib import Path
import numpy as np

from ..TorchModel import Config

class SPEEDplusConfig(Config):
    def __init__(self):
        super().__init__()
        # config
        self.exp_type = "test"
        self.seed = 9999
        self.benchmark = True
        self.debug = True
        self.comet_api = "agcu7oeqU395peWf6NCNqnTa7"
        self.offline = False

        # dataset
        self.dataset = "SPEEDplus"
        self.dataset_folder = Path("../datasets/speedplusv2")
        self.cache = True
        self.resize_first = True
        # self.image_first_size = (900, 1440)
        self.image_first_size = (800, 1280)
        self.image_size = (480, 768)
        # self.image_size = (400, 640)

        # train
        self.device = "cuda"
        self.epochs = 50
        self.batch_size = 50
        self.lr0 = 0.001
        self.lr_min = 0.000001
        self.warmup_epochs = 5
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "WarmupCosin"              # WarmupCosin, OnPlateau, ReduceWarmupCosin, MultiStepLR
        self.num_workers = 30
        self.compile = True
        self.gradient_clip_val = 5.0

        # model
        # backbone
        self.pretrained = True
        self.backbone = "mobilenetv3_large_100"
        self.backbone_args = {
            "mobilenetv3_large_100": dict(),
        }
        # neck
        self.neck = "TailNeck"                  # IdentityNeck, ConvNeck, FPNPAN
        self.neck_args = {
            "TailNeck": {"att_type": None},
            "IdentityNeck": {"out_index": (-1, )},
            "ConvNeck": {"out_index": (-3, -2, -1, )},
            "PAFPN": {"align_channels": 160},
            "BiFPN": {"align_channels": 160},
            "DensAttFPN": {"att_type": None},    # SE, SAM, CBAM, SSIA
        }
        # head
        self.head = "TokenHead"
        self.head_args = {
            "TokenHead": {
                "patch_shape": None,
                "embedding_mode": "mean",
                "num_heads": 8,
                "num_layers": 8,
            }
        }

        # keypoints type
        self.keypoints_type = "heatmap_distribution"
        self.keypoints_args = {
            "heatmap_argmax": {
                "shape_ratio": 1/4,
                "heatmap_ratio": 1/4,
                "sigma": 3.0,
            },
            "heatmap_distribution": {
                "heatmap_ratio": 1/2**5,
            }
        }

        # uncertainty
        self.uncertainty_type = "Ratio"
        self.uncertainty_args = {
            "Rank": {},
            "Ratio": {}
        }

        # loss
        self.keypoints_beta = 1.0
        self.keypoints_weight_strategy = None
        self.keypoints_loss_type = "CE"
        self.keypoints_loss_args = {
            "L2": {"reduction": "none"},
            "CE": {"reduction": "none"},
            "KL": {"reduction": "none"},
        }

        self.uncertainty_beta = 1.0
        self.uncertainty_weight_strategy = None
        self.uncertainty_loss_type = "L2"
        self.uncertainty_loss_args = {
            "L1": {"reduction": "none"},
            "L2": {"reduction": "none"},
            "Sub": {},
        }

        self.ALPHA = (5, 1)              # score

        # augmentation
        self.OpticalCenterRotation_p = 0.8
        self.OpticalCenterRotation_args = {
            "max_angle": 180,
            "max_t": 7,
        }

        self.TransRotation_p = 0.0
        self.TransRotation_args = {
            "max_angle": 5,
            "max_trans_xy": 0.2,
            "max_trans_z": 0.5,
            "max_t": 7,
        }

        self.ClothSurface_p = 0.3

        self.SurfaceBrightness_p = 0.5

        self.SurfaceSunFlare_p = 0.5

        self.SunFlare_p = 0.2

        self.CropAndPaste_p = 0.0

        self.CropAndPadSafe_p = 0.0

        self.DropBlockSafe_p = 0.0
        self.DropBlockSafe_args = {
            "drop_num": 7,
        }

        self.AlbumentationAug_p = 0.05

        self.name = ""

        self.keypoints = np.array([
            [-0.37,   -0.385,   0.3215],
            [-0.37,    0.385,   0.3215],
            [ 0.37,    0.385,   0.3215],
            [ 0.37,   -0.385,   0.3215],
            [-0.37,   -0.264,   0.    ],
            [-0.37,    0.304,   0.    ],
            [ 0.37,    0.304,   0.    ],
            [ 0.37,   -0.264,   0.    ],
            [-0.5427,  0.4877,  0.2535],
            [ 0.5427,  0.4877,  0.2591],
            [ 0.305,  -0.579,   0.2515]
        ])

        # self.keypoints = np.array([
        #     [-0.37,   -0.264,   0.3215],
        #     [-0.37,    0.304,   0.3215],
        #     [ 0.37,    0.304,   0.3215],
        #     [ 0.37,   -0.264,   0.3215],
        #     [-0.37,   -0.264,   0.    ],
        #     [-0.37,    0.304,   0.    ],
        #     [ 0.37,    0.304,   0.    ],
        #     [ 0.37,   -0.264,   0.    ],
        #     [-0.5427,  0.4877,  0.2535],
        #     [ 0.5427,  0.4877,  0.2591],
        #     [ 0.305,  -0.579,   0.2515]
        # ])