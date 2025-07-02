from .heatmap_argmax import HeatmapArgmaxEncoder, HeatmapArgmaxDecoder
from .heatmap_distribution import HeatmapDistributionEncoder, HeatmapDistributionDecoder

def get_keypoints_encoder(keypoints_type: str, **kwargs):
    if keypoints_type == "heatmap_argmax":
        return HeatmapArgmaxEncoder(**kwargs)
    elif keypoints_type == "heatmap_distribution":
        return HeatmapDistributionEncoder(**kwargs)
    else:
        raise ValueError(f"Unsupported keypoints type: {keypoints_type}")

def get_keypoints_decoder(keypoints_type: str, **kwargs):
    if keypoints_type == "heatmap_argmax":
        return HeatmapArgmaxDecoder(**kwargs)
    elif keypoints_type == "heatmap_distribution":
        return HeatmapDistributionDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported keypoints type: {keypoints_type}")