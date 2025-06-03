import torch

from .callback import Callback
from torchinfo import summary

class ModelSummary(Callback):
    def __init__(self,
                 input_size: tuple = (1, 1, 480, 768),
                 col_names = ("input_size",
                              "output_size",
                              "num_params",
                              "params_percent",
                              "mult_adds"),
                 col_width: int = 24,
                 depth: int = 3,
                 ):
        self.input_size = input_size
        self.col_names = col_names
        self.col_width = col_width
        self.depth = depth
    
    def on_fit_start(self, trainer):
        with torch.no_grad():
            result = summary(trainer.model,
                            input_size=self.input_size,
                            col_names=self.col_names,
                            col_width=self.col_width,
                            depth=self.depth)
        model_metrics = {
            "MACs(G)": result.total_mult_adds / 1e9,
            "Params(M)": result.total_params / 1e6,
        }
        trainer.logger.log_dict(model_metrics, epoch=0)
        trainer.logger.log_text(str(result))