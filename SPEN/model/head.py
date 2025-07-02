from .blocks import *
from typing import List, Dict, Any


class BaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.keypoints_feature_dims: List[int]
        self.uncertainty_feature_dims: List[int]
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TokenHead(BaseHead):
    def __init__(
            self,
            in_channels: List[int],
            **kwargs,
    ):
        super().__init__()
        num_heads = kwargs.get("num_heads", 8)
        num_layers = kwargs.get("num_layers", 8)
        patch_shape = kwargs.get("patch_shape", None)
        embedding_mode = kwargs.get("embedding_mode", "mean")
        in_channels = in_channels[-1]

        self.patch_embedding = PatchEmbedding(
            patch_shape=patch_shape,
            embedding_mode=embedding_mode,
        )

        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=in_channels,
                    num_heads=num_heads,
                ) for _ in range(num_layers)
            ]
        )

        self.last_layer_norm = nn.LayerNorm(in_channels)

        self.learnable_token = nn.Parameter(torch.randn(1, 1, in_channels), requires_grad=True)
        max_embedding_len = 500
        self.position_embedding = nn.Parameter(torch.randn(1, max_embedding_len, in_channels) * .02, requires_grad=True)
        self.uncertainty_feature_dims = self.keypoints_feature_dims = in_channels
        self.init_weights()
    

    def add_token(self, patch: Tensor):
        B = patch.shape[0]
        patch = torch.cat([self.learnable_token.expand(B, 1, -1), patch], dim=1)
        return patch
    

    def pos_embedding(self, patch: Tensor):
        S = patch.shape[1]
        patch = patch + self.position_embedding[:, :S, :]
        return patch


    def forward(self, x: List[Tensor]):
        x = x[-1]
        B, C, H, W = x.shape
        patch = self.patch_embedding(x)     # B, S, C
        patch = self.add_token(patch)     # B, S+1, C
        patch = self.pos_embedding(patch)     # B, S+1, C
        out = self.blocks(patch)     # B, S+1, C
        out = self.last_layer_norm(out)     # B, S+1, C
        out = out.transpose(1, 2)     # B, C, S+1
        uncertainty_feature, heatmap_feature = torch.split(out, (1, H*W), dim=-1)
        uncertainty_feature = uncertainty_feature.squeeze(dim=-1)
        heatmap_feature = heatmap_feature.reshape(B, C, H, W)
        return uncertainty_feature, heatmap_feature


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif hasattr(m, "init_weights") and m is not self:
                m.init_weights()
        trunc_normal_(self.position_embedding, std=.02)
        trunc_normal_(self.learnable_token, std=1e-6)


class HeadFactory:

    head_dict = {
        "TokenHead": TokenHead,
    }

    def __init__(self):
        pass

    def create_head(
            self,
            head: str,
            in_channels: List[int],
            **kwargs: Dict[str, Any],
    ):
        HeadClass = HeadFactory.head_dict.get(head, None)
        if HeadClass is None:
            raise ValueError(f"Unsupported head model: {head}.")
        model = HeadClass(
            in_channels=in_channels,
            **kwargs,
        )
        return model
