from torchinfo import summary
from SPEN.model import SPEN
from SPEN.cfg import SPEEDplusConfig
import torch
import time

torch.set_float32_matmul_precision('high')
config = SPEEDplusConfig()
# config.backbone = "mobilenetv3_large_100"
# config.neck = "TailNeck"
# config.head = "TokenHead"
# config.pos_type = "DiscreteSpher"
# config.pos_loss = "CE"
# config.ori_type = "DiscreteEuler"
# config.ori_loss = "CE"
# config.pos_args["DiscreteSpher"]["r_stride"] = 1
# config.pos_args["DiscreteSpher"]["angle_stride"] = 1
# config.ori_type = "DiscreteEuler"
# config.ori_loss_type = "CE"
# config.ori_args["DiscreteEuler"]["stride"] = 1
config.batch_size = 1
config.image_size = (480, 768)
model = SPEN(config=config)
print(model.backbone.model.feature_info.channels())
input_tensor = torch.randn(config.batch_size, 1, *config.image_size)
output = model(input_tensor)
print(output)
result = summary(model,
                input_size=(config.batch_size, 1, *config.image_size),
                col_names=("input_size", "output_size", "num_params", "params_percent", "mult_adds"),
                depth=3)
model_metrics = {
    "MACs(G)": result.total_mult_adds / 1e9,
    "Params(M)": result.total_params / 1e6,
}
print(model_metrics)