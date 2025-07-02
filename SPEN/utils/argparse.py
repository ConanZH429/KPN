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
    parser.add_argument("--num_heads", type=int, default=config.head_args["TokenHead"]["num_heads"])
    parser.add_argument("--num_layers", type=int, default=config.head_args["TokenHead"]["num_layers"])
    # score
    parser.add_argument("--ALPHA", nargs="+", default=config.ALPHA, help="val score alpha")

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
    if config.head in {"TokenHead"}:
        config.head_args[config.head]["num_heads"] = args.num_heads
        config.head_args[config.head]["num_layers"] = args.num_layers
    # score
    config.ALPHA = tuple(map(float, args.ALPHA))
    # name
    config.name = f"{config.exp_type}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"

    return config