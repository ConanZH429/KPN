import timm
import torch.nn as nn
from torch import Tensor

from typing import List


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels: List[int] = None
        self.pretrained: bool = None
        self.model: nn.Module = None
        

class MobilenetV3Large(BaseBackbone):
    def __init__(self,
                 pretrained: bool = True,
                 bin_path: str = None,
                 in_chans: int = 1,
                 features_only: bool = True,
                 **kwargs):
        super().__init__()
        self.model: nn.Module = timm.create_model(
            "mobilenetv3_large_100",
            pretrained=pretrained,
            pretrained_cfg_overlay=dict(file=str(bin_path)),
            in_chans=in_chans,
            features_only=features_only,
        )
        self.model.blocks[-1] = nn.Identity()
        self.out_channels = self.model.feature_info.channels()
        self.out_channels[-1] = 160
    
    def forward(self, x) -> List[Tensor]:
        return self.model(x)



class BackboneFactory:

    backbone_dict = {
        "mobilenetv3_large_100": MobilenetV3Large,
    }

    backbone_bin_folder = {
        "mobilenetv3_large_100": "mobilenetv3_large_100.miil_in21k",
    }

    def __init__(self):
        pass

    def create_backbone(
            self,
            backbone: str,
            pretrained: bool = True,
            **kwargs
    ) -> BaseBackbone:
        BackboneClass = BackboneFactory.backbone_dict.get(backbone, None)
        if not BackboneClass:
            raise ValueError(f"Unsupported backbone model: {backbone}.")
        bin_folder = BackboneFactory.backbone_bin_folder.get(backbone, None)
        if not bin_folder:
            raise ValueError(f"Bin folder not found for model: {backbone}.")
        else:
            bin_path = f"./SPEN/model/timm_weight/{bin_folder}/pytorch_model.bin"
        model = BackboneClass(
            pretrained=pretrained,
            bin_path=bin_path,
            in_chans=1,
            features_only=True,
            **kwargs
        )
        return model