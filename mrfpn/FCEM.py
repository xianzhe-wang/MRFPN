# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.runner import BaseModule, ModuleList

from ..builder import NECKS, build_backbone
from .fpn import FPN


class ASPP(BaseModule):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations=(1, 3, 6, 1),
                 init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])
        out = torch.cat(out, dim=1)
        return out


@NECKS.register_module()
class FCEM(FPN):
    

    def __init__(self,
                 rfp_steps,
                 rfp_backbone,
                 aspp_out_channels,
                 aspp_dilations=(1, 3, 6, 1),
                 init_cfg=None,
                 **kwargs):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.rfp_steps = rfp_steps
        # Be careful! Pretrained weights cannot be loaded when use
        # nn.ModuleList
        self.rfp_modules = ModuleList()
        for rfp_idx in range(1, rfp_steps):
            rfp_module = build_backbone(rfp_backbone)
            self.rfp_modules.append(rfp_module)
        self.rfp_aspp = ASPP(self.out_channels, aspp_out_channels,
                             aspp_dilations)
        self.rfp_weight = nn.Conv2d(
            self.out_channels,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        # self.fusion0 = Fusion_Exp01(256)
        # self.fusion1 = Fusion_Exp01(256)
        # self.fusion2 = Fusion_Exp01(256)
        # self.fusion3 = Fusion_Exp01(256)
        # self.fusion4 = Fusion_Exp01(256)
        # self.fusion_list = [self.fusion0,self.fusion1,self.fusion2,self.fusion3,self.fusion4]

        self.EXP04_0 = EXP04(channel=256)
        # self.EXP04_1 = EXP04(channel=256)
        # self.EXP04_2 = EXP04(channel=256)
        # self.EXP04_3 = EXP04(channel=256)
        # self.EXP04_4 = EXP04(channel=256)
        # self.EXP04_list = [self.EXP04_0,self.EXP04_1,self.EXP04_2,self.EXP04_3,self.EXP04_4]

    def init_weights(self):
        # Avoid using super().init_weights(), which may alter the default
        # initialization of the modules in self.rfp_modules that have missing
        # keys in the pretrained checkpoint.
        for convs in [self.lateral_convs, self.fpn_convs]:
            for m in convs.modules():
                if isinstance(m, nn.Conv2d):
                    xavier_init(m, distribution='uniform')
        for rfp_idx in range(self.rfp_steps - 1):
            self.rfp_modules[rfp_idx].init_weights()
        constant_init(self.rfp_weight, 0)

    def forward(self, inputs):
        inputs = list(inputs)
        assert len(inputs) == len(self.in_channels) + 1  # +1 for input image
        img = inputs.pop(0)
        # FPN forward
        x = super().forward(tuple(inputs))

        for rfp_idx in range(self.rfp_steps - 1):
            rfp_feats = [x[0]] + list(
                self.EXP04_0(x[i]) for i in range(1, len(x)))
                # self.EXP04_list[i](self.rfp_aspp(x[i])) for i in range(1, len(x)))
            x_idx = self.rfp_modules[rfp_idx].rfp_forward(img, rfp_feats)
            # FPN forward
            x_idx = super().forward(x_idx)
            x_new = []
            for ft_idx in range(len(x_idx)):
                # print(self.rfp_weight(x_idx[ft_idx]).shape)
                add_weight = torch.sigmoid(self.rfp_weight(x_idx[ft_idx]))
                # print(add_weight.shape)
                # print(x[ft_idx].shape)
                # exit()
                # print((add_weight * x_idx[ft_idx]).shape, ((1 - add_weight) * x[ft_idx]).shape)
                # temp1 = add_weight * x_idx[ft_idx]
                # temp2 = (1 - add_weight) * x[ft_idx]
                # tempf = self.fusion_list[ft_idx](temp1, temp2)
                # x_new.append(tempf)
                
                x_new.append(add_weight * x_idx[ft_idx] +
                             (1 - add_weight) * x[ft_idx])
            x = x_new
        # exit()

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class EXP04_ATT(nn.Module):
    def __init__(self, channel):
        super(EXP04_ATT, self).__init__()
        self.relu = nn.ReLU(True)

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def upsample_add(*xs):
    y = xs[-1]
    for x in xs[:-1]:
        y = y + F.interpolate(x, size=y.size()[2:], mode="bilinear", align_corners=False)
    return y

class EXP04(nn.Module):
    def __init__(self, channel=64):
        super(EXP04, self).__init__()
        self.h2l_pool = nn.AvgPool2d((2, 2), stride=2)

        self.h2l = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)
        self.h2h = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

        self.mscah = EXP04_ATT(channel)
        self.mscal = EXP04_ATT(channel)

        self.upsample_add = upsample_add
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        # first conv
        x_h = self.h2h(x)
        x_l = self.h2l(self.h2l_pool(x))
        x_h = x_h * self.mscah(x_h)
        x_l = x_l * self.mscal(x_l)
        out = self.upsample_add(x_l, x_h)
        out = self.conv(out)

        return out