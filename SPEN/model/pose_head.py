from .blocks import *

from typing import Dict, Any, List

class DistributionHeatmapHead(nn.Module):
    def __init__(self, in_channels: int, keypoints_num: int, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=keypoints_num,
            kernel_size=1,
            bias=True
        )
    
    def forward(self, heatmap_feature: Tensor):
        return self.conv(heatmap_feature)


class RankUncertaintyHead(nn.Module):
    def __init__(self, in_channels: int, keypoints_num: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_channels,
            out_features=keypoints_num, 
        )
    
    def forward(self, uncertainty_feature: Tensor):
        uncertainty_feature = self.fc(uncertainty_feature)
        uncertainty_feature = F.sigmoid(uncertainty_feature)
        return uncertainty_feature


class RatioUncertaintyHead(nn.Module):
    def __init__(self, in_channels: int, keypoints_num: int, **kwargs):
        super().__init__()
        self.fc = nn.Linear(
            in_features=in_channels,
            out_features=keypoints_num,
        )
    
    def __init__(self, uncertainty_feature: Tensor):
        uncertainty_feature = self.fc(uncertainty_feature)
        uncertainty_feature = F.sigmoid(uncertainty_feature)
        return uncertainty_feature


class KeypointsHeadFactory:

    keypoints_head_dict = {
        "heatmap_distribution": DistributionHeatmapHead
    }

    def __init__(self):
        pass

    def create_keypoints_head(
            self,
            keypoints_type: str,
            keypoints_loss_type: str,
            keypoints_feature_dims: int,
            keypoints_num: int,
            **kwargs
    ):
        KeypointsHead = KeypointsHeadFactory.keypoints_head_dict[keypoints_type]
        keypoints_head = KeypointsHead(keypoints_feature_dims, keypoints_num, **kwargs)
        return keypoints_head



class UncertaintyHeadFactory:

    uncertainty_head_dict = {
        "Rank": RankUncertaintyHead,
        "Ratio": RatioUncertaintyHead,
    }

    def __init__(self):
        pass
    
    def create_uncertainty_head(
            self,
            uncertainty_type: str,
            uncertainty_loss: str,
            uncertainty_feature_dims: int,
            keypoints_num: int,
            **kwargs,
    ):
        UncertaintyHead = UncertaintyHeadFactory.uncertainty_head_dict[uncertainty_type]
        uncertainty_head = UncertaintyHead(uncertainty_feature_dims, keypoints_num, **kwargs)
        return uncertainty_head