from pathlib import Path

from ..TorchModel import Config

class SPEEDplusConfig(Config):
    def __init__(self):
        super().__init__()
        # config
        self.exp_type = "test"
        self.seed = 9999
        self.benchmark = True
        self.debug = False
        self.comet_api = "agcu7oeqU395peWf6NCNqnTa7"
        self.offline = False

        # dataset
        self.dataset = "SPEEDplus"
        self.dataset_folder = Path("../datasets/speedplusv2")
        self.cache = True
        self.resize_first = True
        self.image_first_size = (900, 1440)
        self.image_size = (480, 768)
        # self.image_size = (400, 640)

        # train
        self.device = "cuda"
        self.epochs = 50
        self.batch_size = 50
        self.lr0 = 0.001
        self.lr_min = 0.000001
        self.warmup_epochs = 0
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "MultiStepLR"              # WarmupCosin, OnPlateau, ReduceWarmupCosin, MultiStepLR
        self.num_workers = 30
        self.compile = True
        self.gradient_clip_val = None

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
            "SplitHead": {"pool_size": (1, ),},
            "PoolHead": {"pool_type": "avg",
                         "pool_size": (1, )},
            "TokenHead": {
                "patch_shape": None,
                "embedding_mode": "mean",
                "num_heads": 8,
                "num_layers": 8,
            }
        }
        
        # pos type
        self.pos_type = "DiscreteSpher"
        self.pos_args = {
            "Cart": {},
            "Spher": {},
            "DiscreteSpher": {
                "r_max": 10,
                "r_stride": 1,
                "angle_stride": 1,
                "device": "cuda",
            }
        }

        # ori type
        self.ori_type = "DiscreteEuler"
        self.ori_args = {
            "Quat": {},
            "Euler": {},
            "DiscreteEuler": {
                "stride": 1,
                "device": "cuda"
            },   
        }
        
        # loss
        ## pos_loss
        # self.pos_loss_dict = {
        #     "DiscreteSpher": "CE",
        # }
        self.pos_loss_dict = {
            "DiscreteSpher": "CE",
            "Spher": "L1",
        }
        self.pos_loss_args = {
            "Cart": {
                "loss_type": {
                    "L1": {"reduction": "mean"},
                    "L2": {"reduction": "mean"},
                    "SmoothL1": {"reduction": "mean"},
                },
                "beta": {"x": 1.0, "y": 1.0, "z": 1.0},
                "weight_strategy": {"x": None, "y": None, "z": None},
            },
            "Spher": {
                "loss_type": {
                    "L1": {"reduction": "mean"},
                    "L2": {"reduction": "mean"},
                    "SmoothL1": {"reduction": "mean"},
                },
                "beta": {"r": 1.0, "theta": 1.0, "phi": 1.0},
                "weight_strategy": {"r": "CosDecay", "theta": "CosDecay", "phi": "CosDecay"},
            },
            "DiscreteSpher": {
                "loss_type": {
                    "CE": {"reduction": "mean"},
                    "KL": {},
                },
                "beta": {"discrete_r": 1.0, "discrete_theta": 1.0, "discrete_phi": 1.0},
                "weight_strategy": {"discrete_r": None, "discrete_theta": None, "discrete_phi": None},
            },
        }
        
        ## ori_loss
        self.ori_loss_dict = {
            "DiscreteEuler": "CE",
            "Euler": "L1",
        }
        self.ori_loss_args = {
            "Quat": {
                "loss_type": {
                    "Cos": {},
                    "CosDistance": {},
                    "ExpCos": {},
                },
                "beta": 5.0,
                "weight_strategy": None,
            },
            "Euler": {
                "loss_type": {
                    "L1": {"reduction": "mean"},
                    "L2": {"reduction": "mean"},
                    "SmoothL1": {"reduction": "mean"},
                },
                "beta": {"yaw": 5.0, "pitch": 5.0, "roll": 5.0},
                "weight_strategy": {"yaw": "CosDecay", "pitch": "CosDecay", "roll": "CosDecay"},
            },
            "DiscreteEuler": {
                "loss_type": {
                    "CE": {"reduction": "mean"},
                    "KL": {},
                },
                "beta": {"discrete_yaw": 5.0, "discrete_pitch": 5.0, "discrete_roll": 5.0},
                "weight_strategy": {"discrete_yaw": None, "discrete_pitch": None, "discrete_roll": None},
            },
        }
        self.ALPHA = (5, 1)              # score

        # augmentation
        self.ZAxisRotation_p = 0.8
        self.ZAxisRotation_args = {
            "max_angle": 180,
            "max_t": 7,
        }

        self.OpticalCenterRotation_p = 0.0
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

        self.ClothSurface_p = 0.0

        self.SurfaceBrightness_p = 0.5

        self.SunFlare_p = 0.0

        self.CropAndPaste_p = 0.0

        self.CropAndPadSafe_p = 0.0

        self.DropBlockSafe_p = 0.0
        self.DropBlockSafe_args = {
            "drop_num": 7,
        }

        self.AlbumentationAug_p = 0.0

        self.name = ""