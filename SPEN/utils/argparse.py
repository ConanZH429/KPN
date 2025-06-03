import argparse
import time
from ..cfg import SPEEDConfig

def parse2config(config):
    parser = argparse.ArgumentParser()
    
    # exp_type
    parser.add_argument("--exp_type", type=str, default="test", help="Experiment type")
    # seed
    parser.add_argument("--seed", type=int, default=config.seed, help="Random seed")
    # dataset
    if isinstance(config, SPEEDConfig):
        parser.add_argument("--train_ratio", type=float, default=config.train_ratio, help="Train ratio")
        parser.add_argument("--val_ratio", type=float, default=config.val_ratio, help="Validation ratio")
    parser.add_argument("--cache", action="store_true", help="Cache dataset")
    parser.add_argument("--resize_first", action="store_true", help="Resize first")
    parser.add_argument("--img_first_size", type=int, nargs="+", default=config.image_first_size, help="Image first size")
    parser.add_argument("--img_size", type=int, nargs="+", default=config.image_size, help="Image size")
    # train
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--lr0", type=float, default=config.lr0, help="Initial learning rate")
    parser.add_argument("--lr_min", type=float, default=config.lr_min, help="Minimum learning rate")
    parser.add_argument("--optimizer", type=str, default=config.optimizer, help="Optimizer")
    parser.add_argument("--scheduler", type=str, default=config.scheduler, help="Scheduler")
    parser.add_argument("--num_workers", type=int, default=config.num_workers, help="Number of workers")
    parser.add_argument("--compile", action="store_true", help="Compile")
    parser.add_argument("--gradient_clip_val", type=float, default=config.gradient_clip_val, help="Gradient clip value")
    # backbone
    parser.add_argument("--backbone", "-b", type=str, default=config.backbone, help="Backbone",
                        choices=list(config.backbone_args.keys()))
    # neck
    parser.add_argument("--neck", "-n", type=str, default=config.neck, help="Neck",
                        choices=list(config.neck_args.keys()))
    parser.add_argument("--align_channels", type=int, default=160, help="Align channels")
    parser.add_argument("--att_type", type=str, default=None, help="Attention type",
                        choices=["SSIA", "SE", "SAM", "CBAM"])
    parser.add_argument("--out_index", type=int, nargs="+", default=(1, ), help="Output index")
    # head
    parser.add_argument("--head", type=str, default=config.head, help="Head",
                        choices=list(config.head_args.keys()))
    parser.add_argument("--pool_size", type=int, nargs="+", default=(1, ), help="Pool size")
    parser.add_argument("--pool_type", type=str, default=config.head_args["PoolHead"]["pool_type"], help="Pool type")
    parser.add_argument("--embed_mode", type=str, default=config.head_args["TokenHead"]["embedding_mode"], help="Embedding mode",)
    parser.add_argument("--num_heads", type=int, default=config.head_args["TokenHead"]["num_heads"])
    parser.add_argument("--num_layers", type=int, default=config.head_args["TokenHead"]["num_layers"])
    # pos
    parser.add_argument("--pos_type", "-pt", type=str, default=config.pos_type, help="Position type",
                        choices=list(config.pos_args.keys()))
    parser.add_argument("--r_max", "-rm", type=int, default=config.pos_args["DiscreteSpher"]["r_max"], help="r max")
    parser.add_argument("--r_stride", "-rs", type=int, default=config.pos_args["DiscreteSpher"]["r_stride"], help="r stride")
    parser.add_argument("--angle_stride", "-as", type=int, default=config.pos_args["DiscreteSpher"]["angle_stride"], help="angle stride")
    # # pos loss
    parser.add_argument("--pos_type4loss", type=str, nargs="+", default=tuple(config.pos_loss_dict.keys()), help="Position type for loss",)
    parser.add_argument("--pos_loss_type", "-plt", type=str, nargs="+", default=tuple(config.pos_loss_dict.values()), help="Position loss type")
    # # ori
    parser.add_argument("--ori_type", "-ot", type=str, default=config.ori_type, help="Orientation type",
                        choices=list(config.ori_args.keys()))
    parser.add_argument("--euler_stride", "-es", type=int, default=config.ori_args["DiscreteEuler"]["stride"], help="stride")
    # # ori loss
    parser.add_argument("--ori_type4loss", type=str, nargs="+", default=tuple(config.ori_loss_dict.keys()), help="Orientation type for loss")
    parser.add_argument("--ori_loss_type", "-olt", type=str, nargs="+", default=tuple(config.ori_loss_dict.values()), help="Orientation loss type")
    # score
    parser.add_argument("--ALPHA", nargs="+", default=config.ALPHA, help="val score alpha")

    # data augmentation
    parser.add_argument("--zr_p", type=float, default=config.ZAxisRotation_p, help="Z axis rotation probability")
    parser.add_argument("--zr_angle", type=int, default=config.ZAxisRotation_args["max_angle"], help="Z axis rotation angle")
    parser.add_argument("--crop_paste_p", type=float, default=config.CropAndPaste_p, help="Crop and paste probability")
    parser.add_argument("--crop_pad_p", type=float, default=config.CropAndPadSafe_p, help="Crop and pad safe probability")
    parser.add_argument("--drop_block_p", type=float, default=config.DropBlockSafe_p, help="Drop block safe probability")
    parser.add_argument("--drop_block_num", type=int, default=config.DropBlockSafe_args["drop_num"], help="Drop block safe number of blocks")
    parser.add_argument("--album_p", type=float, default=config.AlbumentationAug_p, help="Albumentations probability")

    args = parser.parse_args()

    # exp_type
    config.exp_type = args.exp_type
    # seed
    config.seed = args.seed
    # dataset
    if isinstance(config, SPEEDConfig):
        config.train_ratio = args.train_ratio
        config.val_ratio = args.val_ratio
    config.cache = args.cache
    config.resize_first = args.resize_first
    config.image_first_size = tuple(map(int, args.img_first_size))
    config.image_size = tuple(map(int, args.img_size))
    # train
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr0 = args.lr0
    config.lr_min = args.lr_min
    config.optimizer = args.optimizer
    config.scheduler = args.scheduler
    config.num_workers = args.num_workers
    config.compile = args.compile
    config.gradient_clip_val = args.gradient_clip_val
    # backbone
    config.backbone = args.backbone
    # neck
    config.neck = args.neck
    if config.neck in {"PAFPN", "BiFPN"}:
        config.neck_args[config.neck]["align_channels"] = args.align_channels
    if config.neck in {"DensAttFPN", "TailNeck"}:
        config.neck_args[config.neck]["att_type"] = args.att_type
    if config.neck in {"ConvNeck", "IdentityNeck"}:
        config.neck_args[config.neck]["out_index"] = tuple(map(int, args.out_index))
    # head
    config.head = args.head
    if config.head in {"PoolHead", "SplitHead"}:
        config.head_args[config.head]["pool_size"] = tuple(map(int, args.pool_size))
    if config.head in {"PoolHead"}:
        config.head_args[config.head]["pool_type"] = args.pool_type
    if config.head in {"TokenHead"}:
        config.head_args[config.head]["embedding_mode"] = args.embed_mode
        config.head_args[config.head]["num_heads"] = args.num_heads
        config.head_args[config.head]["num_layers"] = args.num_layers
    # pos
    config.pos_type = args.pos_type
    if config.pos_type in {"DiscreteSpher"}:
        config.pos_args[config.pos_type]["r_max"] = args.r_max
        config.pos_args[config.pos_type]["r_stride"] = args.r_stride
        config.pos_args[config.pos_type]["angle_stride"] = args.angle_stride
    # pos loss
    pos_type4loss = tuple(map(str, args.pos_type4loss))
    pos_loss_type = tuple(map(str, args.pos_loss_type))
    config.pos_loss_dict = {
        pos_type: loss_type for pos_type, loss_type in zip(pos_type4loss, pos_loss_type)
    }
    # ori
    config.ori_type = args.ori_type
    if config.ori_type in {"DiscreteEuler"}:
        config.ori_args[config.ori_type]["stride"] = args.euler_stride
    # ori loss
    ori_type4loss = tuple(map(str, args.ori_type4loss))
    ori_loss_type = tuple(map(str, args.ori_loss_type))
    config.ori_loss_dict = {
        ori_type: loss_type for ori_type, loss_type in zip(ori_type4loss, ori_loss_type)
    }
    # score
    config.ALPHA = tuple(map(float, args.ALPHA))
    # data augmentation
    config.ZAxisRotation_p = args.zr_p
    config.ZAxisRotation_args["max_angle"] = args.zr_angle
    config.CropAndPaste_p = args.crop_paste_p
    config.CropAndPadSafe_p = args.crop_pad_p
    config.DropBlockSafe_p = args.drop_block_p
    config.DropBlockSafe_args["drop_num"] = args.drop_block_num
    config.AlbumentationAug_p = args.album_p
    # name
    config.name = f"{config.exp_type}_{config.backbone}_{config.neck}_{config.pos_type}_{config.ori_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    return config