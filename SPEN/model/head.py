from .blocks import *
from typing import List, Dict, Any


class BaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_feature_dims: List[int]
        self.ori_feature_dims: List[int]
    
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


class SplitHead(BaseHead):
    def __init__(
            self,
            in_channels: List[int],
            **kwargs,
    ):
        super().__init__()
        self.pool_size = kwargs.get("pool_size", None)
        self.fuse = kwargs.get("fuse", True)
        self.pos_ratio = kwargs.get("pos_ratio", 0.2)
        if isinstance(self.pool_size[0], int):
            self.pool_size = [(ps, ps) for ps in self.pool_size]
        feature_dim = sum(channels * ps[0]*ps[1] for channels, ps in zip(in_channels, self.pool_size))
        self.fuse_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            MLPAct(inplace=True)
        ) if self.fuse else nn.Identity()
        self.pos_feature_dims = [int(feature_dim * self.pos_ratio)]
        self.ori_feature_dims = [feature_dim - int(feature_dim * self.pos_ratio)]
        self.init_weights()

    def forward(self, x: List[Tensor]):
        x = x[-len(self.pool_size):]
        x_pooled = [F.adaptive_avg_pool2d(fm, ps).flatten(1) for fm, ps in zip(x, self.pool_size)]
        feature = torch.cat(x_pooled, dim=1)
        feature = self.fuse_fc(feature)
        pos_feature, ori_feature = torch.split(feature, [self.pos_feature_dims[0], self.ori_feature_dims[0]], dim=1)
        return pos_feature, ori_feature


class PoolHead(BaseHead):
    def __init__(
            self,
            in_channels: List[int],
            **kwargs,
    ):
        super().__init__()
        self.pool_size = kwargs.get("pool_size", None)
        self.pool_type = kwargs.get("pool_type", "avg")
        if self.pool_type == "mix":
            self.weight_learnable = kwargs.get("weight_learnable", False)
            if self.weight_learnable:
                self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32, requires_grad=True), requires_grad=True)
        if isinstance(self.pool_size[0], int):
            self.pool_size = [(ps, ps) for ps in self.pool_size]
        if self.pool_type == "avg":
            self.pos_pool = AvgPool(self.pool_size)
            self.ori_pool = AvgPool(self.pool_size)
        elif self.pool_type == "max":
            self.pos_pool = MaxPool(self.pool_size)
            self.ori_pool = MaxPool(self.pool_size)
        elif self.pool_type == "mix":
            self.pos_pool = MixPool(self.pool_size)
            self.ori_pool = MixPool(self.pool_size)
        else:
            raise ValueError(f"Unsupported pool type: {self.pool_type}.")
        feature_dim = sum(channels * ps[0]*ps[1] for channels, ps in zip(in_channels, self.pool_size))
        if self.pool_type == "mix":
            self.pos_feature_dims = self.ori_feature_dims = [feature_dim * 2]
        else:
            self.pos_feature_dims = self.ori_feature_dims = [feature_dim]
        self.init_weights()
    
    def forward(self, x: List[Tensor]):
        x = x[-len(self.pool_size):]
        pos_feature = self.pos_pool(x)
        ori_feature = self.ori_pool(x)
        return pos_feature, ori_feature


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

        self.learnable_token = nn.Parameter(torch.zeros(1, 6, in_channels), requires_grad=True)
        max_embedding_len = 1000
        self.position_embedding = nn.Parameter(torch.randn(1, max_embedding_len, in_channels) * .02, requires_grad=True)
        self.pos_feature_dims = self.ori_feature_dims = [in_channels, in_channels, in_channels]
        self.init_weights()
    

    def add_token(self, patch: Tensor):
        B = patch.shape[0]
        patch = torch.cat([self.learnable_token.expand(B, -1, -1), patch], dim=1)
        return patch
    

    def pos_embedding(self, patch: Tensor):
        S = patch.shape[1]
        patch = patch + self.position_embedding[:, :S, :]
        return patch


    def forward(self, x: List[Tensor]):
        x = x[-1]
        B, C, H, W = x.shape
        patch = self.patch_embedding(x)     # B, S, C
        patch = self.add_token(patch)     # B, S+6, C
        patch = self.pos_embedding(patch)     # B, S+6, C
        out = self.blocks(patch)     # B, S+6, C
        out = self.last_layer_norm(out)     # B, S+6, C
        out = out.transpose(1, 2)     # B, C, S+6
        pos_feature = out[:, :, :3]     # B, C, 3
        ori_feature = out[:, :, 3:6]    # B, C, 3
        return pos_feature, ori_feature


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
        "PoolHead": PoolHead,
        "TokenHead": TokenHead,
        "SplitHead": SplitHead,
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
