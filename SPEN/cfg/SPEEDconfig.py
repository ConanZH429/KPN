from pathlib import Path

from ..TorchModel import Config

class SPEEDConfig(Config):
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
        self.dataset = "SPEED"
        self.dataset_folder = Path("../datasets/speed")
        self.train_ratio = 0.85
        self.val_ratio = 0.15
        self.cache = False
        self.resize_first = True
        self.image_first_size = (1000, 1600)
        # self.image_first_size = (800, 1280)
        # self.image_size = (480, 768)
        self.image_size = (400, 640)

        # train
        self.device = "cuda"
        self.epochs = 10
        self.batch_size = 1
        self.lr0 = 0.001
        self.lr_min = 0.000001
        self.warmup_epochs = 5
        self.weight_decay = 0.00001
        self.optimizer = "AdamW"
        self.scheduler = "WarmupCosin"              # WarmupCosin, OnPlateau, ReduceWarmupCosin
        self.num_workers = 4
        self.compile = False
        self.gradient_clip_val = None

        # model
        # backbone
        self.pretrained = True
        self.backbone = "mobilenetv3_large_100"
        self.backbone_args = {
            "mobilenetv3_large_100": dict(),
        }
        # self.backbone_args = {
        #     "mobilenetv3_small_075": {
        #         "bin_folder": "mobilenetv3_small_075.lamb_in1k",
        #         "out_channels": [16, 24, 32, 40, 72],
        #     },
        #     "mobilenetv3_small_100": {
        #         "bin_folder": "mobilenetv3_small_100.lamb_in1k",
        #         "out_channels": [16, 24, 40, 48, 96]
        #     },
        #     "mobilenetv3_large_075": {
        #         "bin_folder": "tf_mobilenetv3_large_075.in1k",
        #         "out_channels": [16, 24, 32, 64, 88, 120]
        #     },
        #     "mobilenetv3_large_100": {
        #         "bin_folder": "mobilenetv3_large_100.miil_in21k",
        #         "out_channels": [16, 24, 40, 80, 112, 160]
        #     },
        #     "mobilenetv3_large_150d": {
        #         "bin_folder": "mobilenetv3_large_150d.ra4_e3600_r256_in1k",
        #         "out_channels": [24, 40, 64, 120, 168, 240]
        #     },
        #     "resnet34": {
        #         "bin_folder": "resnet34.a1_in1k",
        #         "out_channels": [64, 64, 128, 256, 512]
        #     },
        #     "efficientnet_b3": {
        #         "bin_folder": "efficientnet_b3.ra2_in1k",
        #         "out_channels": [24, 32, 48, 136, 384]
        #     }
        # }
        # self.WMSA = False
        # self.GMMSA = False
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
                "r_max": 50,
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
                "weight_strategy": {"r": None, "theta": None, "phi": None},
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
                "weight_strategy": {"yaw": None, "pitch": None, "roll": None},
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
        self.ZAxisRotation_p = 0.0
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

        self.SurfaceBrightness_p = 0.0

        self.SunFlare_p = 0.0

        self.CropAndPaste_p = 0.0

        self.CropAndPadSafe_p = 0.0

        self.DropBlockSafe_p = 0.0
        self.DropBlockSafe_args = {
            "drop_num": 7,
        }

        self.AlbumentationAug_p = 0.0

        self.name = ""