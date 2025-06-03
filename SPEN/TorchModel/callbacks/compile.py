import torch

from .callback import Callback

class Compile(Callback):
    def __init__(self,
                 mode: str,
                 fullgraph: bool):
        super().__init__()
        self.mode = mode
        self.fullgraph = fullgraph
    
    def on_fit_start(self, trainer):
        trainer.model.model = torch.compile(trainer.model.model,
                                            mode=self.mode,
                                            fullgraph=self.fullgraph)