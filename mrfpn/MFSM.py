# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn.functional as F
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import NonLocal2d
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class MFSM(BaseModule):


    def __init__(self,
                 in_channels,
                 num_levels,
                 refine_level=2,
                 refine_type=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(MFSM, self).__init__(init_cfg)
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.refine_level = refine_level
        self.refine_type = refine_type
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
            
        # self.fusion_0 = Fusion_Exp02(256)
        # self.fusion_1 = Fusion_Exp02(256)
        # self.fusion_2 = Fusion_Exp02(256)
        # self.fusion_3 = Fusion_Exp02(256)
        # self.fusion_4 = Fusion_Exp02(256)
        # self.fusion_list = [self.fusion_0,self.fusion_1,self.fusion_2,self.fusion_3,self.fusion_4]
        self.FE0 = FeatureExtractionExp03(256)
        self.FE1 = FeatureExtractionExp03(256)
        self.FE2 = FeatureExtractionExp03(256)
        self.FE3 = FeatureExtractionExp03(256)
        self.FE4 = FeatureExtractionExp03(256)
        self.FE_list = [self.FE0,self.FE1,self.FE2,self.FE3,self.FE4]

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == self.num_levels

        # step 1: gather multi-level features by resize and average
        feats = []
        gather_size = inputs[self.refine_level].size()[2:]
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered = F.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size)
            else:
                gathered = F.interpolate(
                    inputs[i], size=gather_size, mode='nearest')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].size()[2:]
            if i < self.refine_level:
                residual = F.interpolate(bsf, size=out_size, mode='nearest')
            else:
                residual = F.adaptive_max_pool2d(bsf, output_size=out_size)
            
            # o = self.fusion_list[i](residual, inputs[i])
            # outs.append(o)

            outs.append(self.FE_list[i](residual + inputs[i]))

            # outs.append(residual + inputs[i])

        return tuple(outs)



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, bn=True, relu=False):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes) if bn else False
        self.relu = nn.ReLU(inplace=True) if relu else False

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.relu:
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

class Fusion_Exp01(nn.Module):
    def __init__(self, channel=64):
        super(Fusion_Exp01, self).__init__()

        self.conv1 = BasicConv2d(channel * 2, channel, kernel_size=3, stride=1, padding=1, relu=True)
        # self.msca = MSCA(channels=channel)
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.conv = BasicConv2d(channel, channel, kernel_size=3, stride=1, padding=1, relu=True)

    def forward(self, x, y):

        xy = torch.cat((x, y), dim=1)
        xy = self.conv1(xy)
        xy = self.ca(xy) * xy
        xy = self.sa(xy) * xy

        # x = xy + x + y
        x = x + xy
        y = y + xy

        x = x * y
        x = self.conv(x)

        return x

class Fusion_Exp02(nn.Module):
    def __init__(self, channel=64):
        super(Fusion_Exp02, self).__init__()

        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()

    def forward(self, att, x):

        x = (self.ca(x) + att) * x
        x = (self.sa(x) + att) * x

        return x

class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class FeatureExtractionExp03(nn.Module):
    def __init__(self, channel=64):
        super(FeatureExtractionExp03, self).__init__()

        self.conv1_1 = nn.Conv2d(channel, channel,
                              kernel_size=1, stride=1,
                              padding=0, dilation=1, bias=False)
        self.conv1_2 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=1, dilation=1, bias=False)
        self.conv1_3 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=3, dilation=3, bias=False)
        
        self.conv2_1 = nn.Conv2d(channel, channel,
                              kernel_size=1, stride=1,
                              padding=0, dilation=1, bias=False)
        self.conv2_2 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=1, dilation=1, bias=False)
        self.conv2_3 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=5, dilation=5, bias=False)
        
        self.conv3_1 = nn.Conv2d(channel, channel,
                              kernel_size=1, stride=1,
                              padding=0, dilation=1, bias=False)
        self.conv3_2 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=1, dilation=1, bias=False)
        self.conv3_3 = nn.Conv2d(channel, channel,
                              kernel_size=3, stride=1,
                              padding=7, dilation=7, bias=False)
        
        self.conv_cat = nn.Conv2d(channel*3, channel,
                              kernel_size=1, stride=1,
                              padding=0, dilation=1, bias=False)
        
        self.att = SimAM()
    
    def forward(self, x):
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)

        x2 = self.conv2_1(x) + x1
        x2 = self.conv2_2(x2)
        x2 = self.conv2_3(x2)

        x3 = self.conv3_1(x) + x2
        x3 = self.conv3_2(x3)
        x3 = self.conv3_3(x3)

        x_all = torch.cat((x1,x2,x3), dim=1)
        x_all = self.conv_cat(x_all)

        return self.att(x) + x_all + x

    