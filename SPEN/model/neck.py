from .blocks import *
from typing import Optional, List


class BaseNeck(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_channels: List[int] = None


class TailNeck(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        self.out_channels = [in_channels[-1]]
        self.att_type = kwargs.get("att_type", None)

        if self.att_type is None:
            self.att = nn.Identity()
        elif self.att_type == "SE":
            rd_ratio = kwargs.get("rd_ratio", 1/4)
            self.att = SEModule(channels=in_channels[-1], rd_ratio=rd_ratio)
        else:
            raise ValueError(f"Unsupported attention type: {self.att_type}.")
        
    def forward(self, x: List[Tensor]):
        return [self.att(x[-1])]


class IdentityNeck(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        self.out_index = kwargs.get("out_index", (-3, -2, -1))
        self.out_channels = [in_channels[i] for i in self.out_index]
        
    def forward(self, x: List[Tensor]):
        return [x[i] for i in self.out_index]


class ConvNeck(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        self.out_index = kwargs.get("out_index", (-3, -2, -1))
        self.out_channels = [in_channels[i] for i in self.out_index]
        self.conv_list = nn.ModuleList(
            [
                ConvNormAct(in_channels[i], in_channels[i], 3, act_layer=ConvAct)
                for i in self.out_index
            ]
        )
        self.feature_len = len(self.out_index)
    
    def forward(self, x: List[Tensor]):
        return [self.conv_list[i](x[self.out_index[i]]) for i in range(self.feature_len)]


class PAFPN(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        align_channels = kwargs.get("align_channels", 160)
        self.out_channels = [align_channels, align_channels, align_channels]
        self.align_p3 = ConvNormAct(in_channels[-3], align_channels, 1, act_layer=ConvAct)
        self.align_p4 = ConvNormAct(in_channels[-2], align_channels, 1, act_layer=ConvAct)
        self.align_p5 = ConvNormAct(in_channels[-1], align_channels, 1, act_layer=ConvAct)

        # FPN
        self.conv5_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.conv4_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.conv3_up = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)

        # PAN
        self.conv3_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.downsample_p3 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv4_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)
        self.downsample_p4 = ConvNormAct(align_channels, align_channels, 3, stride=2, apply_act=False, apply_norm=False)
        self.conv5_down = ConvNormAct(align_channels, align_channels, 3, act_layer=ConvAct)   
    
    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal FPNPAN unit
            P5_0 ---------> P5_1 ---------> P5_2 -------->
                             |                ↑
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
                             |                ↑
                             ↓                |
            P3_0 ---------> P3_1 ---------> P3_2 -------->
        """
        # align
        p3_0 = self.align_p3(x[-3])
        p4_0 = self.align_p4(x[-2])
        p5_0 = self.align_p5(x[-1])

        # FPN
        p5_1 = self.conv5_up(p5_0)
        p4_1 = self.conv4_up(p4_0 + F.interpolate(p5_1, size=p4_0.shape[-2:], mode="nearest"))
        p3_1 = self.conv3_up(p3_0 + F.interpolate(p4_1, size=p3_0.shape[-2:], mode="nearest"))

        # PAN
        p3_2 = self.conv3_down(p3_1)
        p4_2 = self.conv4_down(p4_1 + self.downsample_p3(p3_2))
        p5_2 = self.conv5_down(p5_1 + self.downsample_p4(p4_2))

        return p3_2, p4_2, p5_2



class BiFPN(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        align_channels = kwargs.get("align_channels", 160)
        self.out_channels = [align_channels, align_channels, align_channels]
        self.eps = torch.tensor(1e-6, dtype=torch.float32)
        # align conv
        self.align_p3 = ConvNormAct(in_channels[-3], align_channels, kernel_size=1, apply_act=False, bias=True)
        self.align_p4 = ConvNormAct(in_channels[-2], align_channels, kernel_size=1, apply_act=False, bias=True)
        self.align_p5 = ConvNormAct(in_channels[-1], align_channels, kernel_size=1, apply_act=False, bias=True)

        # conv layers
        self.conv4_up = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv3_up = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv4_down = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )
        self.conv5_down = nn.Sequential(
            ConvNormAct(align_channels, align_channels, kernel_size=3, groups=align_channels, apply_act=False, apply_norm=False),
            ConvNormAct(align_channels, align_channels, kernel_size=1, act_layer=nn.SiLU)
        )

        # weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)

    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal bifpn unit
            P5_0 -------------------------> P5_2 -------->
               |                              ↑
               |-------------|                |
                             ↓                |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
               |             |              ↑ ↑
               |-------------+--------------| |
                             |                |
                             |--------------| |
                                            ↓ |
            P3_0 -------------------------> P3_2 -------->
        """
        # align
        p3_0 = self.align_p3(x[-3])
        p4_0 = self.align_p4(x[-2])
        p5_0 = self.align_p5(x[-1])

        # weights for P5_0 and P4_0 to P4_1
        # p4_w1 = F.relu(self.p4_w1)
        # weight = p4_w1 / (torch.sum(p4_w1) + self.eps)
        weight = F.softmax(self.p4_w1, dim=0)
        p4_1 = self.conv4_up(
            weight[0] * p4_0 + weight[1] * F.interpolate(p5_0, size=p4_0.shape[-2:], mode="bilinear")
        )

        # weights for P4_1 and P3_0 to P3_2
        # p3_w2 = F.relu(self.p3_w2)
        # weight = p3_w2 / (torch.sum(p3_w2) + self.eps)
        weight = F.softmax(self.p3_w2, dim=0)
        p3_2 = self.conv3_up(
            weight[0] * p3_0 + weight[1] * F.interpolate(p4_1, size=p3_0.shape[-2:], mode="bilinear")
        )
        
        # weights for P4_0, P4_1 and P3_0 to P4_2
        # p4_w2 = F.relu(self.p4_w2)
        # weight = p4_w2 / (torch.sum(p4_w2) + self.eps)
        weight = F.softmax(self.p4_w2, dim=0)
        p4_2 = self.conv4_down(
            weight[0] * p4_0 + weight[1] * p4_1 + weight[2] * F.max_pool2d(F.interpolate(p3_2, size=(p4_1.shape[-2]*2, p4_1.shape[-1]*2), mode="bilinear"), kernel_size=2, stride=2)
        )

        # weights for P5_0 and P4_2 to P5_2
        # p5_w2 = F.relu(self.p5_w2)
        # weight = p5_w2 / (torch.sum(p5_w2) + self.eps)
        weight = F.softmax(self.p5_w2, dim=0)
        p5_2 = self.conv5_down(
            weight[0] * p5_0 + weight[1] * F.max_pool2d(F.interpolate(p4_2, size=(p5_0.shape[-2]*2, p5_0.shape[-1]*2), mode="bilinear"), kernel_size=2, stride=2)
        )

        return p3_2, p4_2, p5_2


class DensAttFPN(BaseNeck):
    def __init__(self, in_channels: List[int], **kwargs):
        super().__init__()
        self.out_channels = in_channels[-3:]
        att_type = kwargs.get("att_type", None)
        # FPN
        self.fuse4_up = AttFuse(in_channels[-3:], att_type)
        self.conv4_up = InvertedResidual(sum(in_channels[-3:]), in_channels[-2], exp_ratio=8, act_layer=ConvAct)
        # self.conv4_up = UniversalInvertedResidual(sum(in_channels[-3:]), in_channels[-2],
        #                                           exp_ratio=8, act_layer=ConvAct, layer_scale_init_value=None)
        self.fuse3_up = AttFuse(in_channels[-4:-1], att_type)
        self.conv3_up = InvertedResidual(sum(in_channels[-4:-1]), in_channels[-3], exp_ratio=8, act_layer=ConvAct)
        # self.conv3_up = UniversalInvertedResidual(sum(in_channels[-4:-1]), in_channels[-3],
        #                                           exp_ratio=8, act_layer=ConvAct, layer_scale_init_value=None)

        # PAN
        self.downsample_p3 = ConvNormAct(in_channels[-3], in_channels[-3], 3, stride=2, apply_act=False, apply_norm=False)
        self.conv4_down = InvertedResidual(sum(in_channels[-3:-1]), in_channels[-2], exp_ratio=8, act_layer=ConvAct)
        # self.conv4_down = UniversalInvertedResidual(sum(in_channels[-3:-1]), in_channels[-2],
        #                                             exp_ratio=8, act_layer=ConvAct, layer_scale_init_value=None)
        self.downsample_p4 = ConvNormAct(in_channels[-2], in_channels[-2], 3, stride=2, apply_act=False, apply_norm=False)
        self.conv5_down = InvertedResidual(sum(in_channels[-2:]), in_channels[-1], exp_ratio=8, act_layer=ConvAct)
        # self.conv5_down = UniversalInvertedResidual(sum(in_channels[-2:]), in_channels[-1],
        #                                             exp_ratio=8, act_layer=ConvAct, layer_scale_init_value=None)

        # self.out_channels = [in_channels[-1]]
        self.init_weights()
    
    def forward(self, x: List[Tensor]):
        """
        illustration of a minimal FPNPAN unit
            P5_0 -------------------------> P5_2 -------->
              |-------------|                 ↑
                            ↓                 |
            P4_0 ---------> P4_1 ---------> P4_2 -------->
                            ↑ |               ↑
                       |----| |               |
              |--------|      ↓               |
            P3_0 ---------> P3_1 -------------|---------->
                            ↑
                       |----|
              |--------|
            P2_0
        """
        # align
        p2_0, p3_0, p4_0, p5_0 = x[-4:]

        # FPN
        p4_1 = self.conv4_up(self.fuse4_up(p3_0, p4_0, p5_0))
        p3_1 = self.conv3_up(self.fuse3_up(p2_0, p3_0, p4_0))

        # PAN
        # p4_2 = self.conv4_down(p4_1 + self.downsample_p3(p3_1))
        # p5_2 = self.conv5_down(p5_0 + self.downsample_p4(p4_2))
        p4_2 = self.conv4_down(torch.cat([p4_1, self.downsample_p3(p3_1)], dim=1))
        p5_2 = self.conv5_down(torch.cat([p5_0, self.downsample_p4(p4_2)], dim=1))

        return p3_1, p4_2, p5_2
        # return [p5_2]

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


class NeckFactory:

    neck_dict = {
        "TailNeck": TailNeck,
        "IdentityNeck": IdentityNeck,
        "ConvNeck": ConvNeck,
        "PAFPN": PAFPN,
        "BiFPN": BiFPN,
        "DensAttFPN": DensAttFPN,
    }

    def __init__(self):
        pass

    def create_neck(
            self,
            neck: str,
            in_channels: List[int],
            **kwargs
    ):
        NeckClass = NeckFactory.neck_dict.get(neck, None)
        if not NeckClass:
            raise ValueError(f"Unsupported neck model: {neck}.")
        model = NeckClass(
            in_channels=in_channels,
            **kwargs
        )
        return model