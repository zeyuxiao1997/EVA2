#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   arch_util.py
@Time    :   2021/05/07 20:52:33
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from skimage import morphology
import numpy as np


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def conv(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(0.1, inplace=True)
    )


def upconv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output


def resnet_block(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock(in_channels, kernel_size, dilation, bias=bias)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out



def resnet_block_w_BN(in_channels,  kernel_size=3, dilation=[1,1], bias=True):
    return ResnetBlock_w_BN(in_channels, kernel_size, dilation, bias=bias)


class ResnetBlock_w_BN(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(ResnetBlock_w_BN, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[0], padding=((kernel_size-1)//2)*dilation[0], bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, dilation=dilation[1], padding=((kernel_size-1)//2)*dilation[1], bias=bias),
            nn.BatchNorm2d(in_channels),
        )
    def forward(self, x):
        out = self.stem(x) + x
        return out


class EventModulation_AlignSeg(nn.Module):
    def __init__(self, nf):
        super(EventModulation_AlignSeg, self).__init__()

        self.nf = nf
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(self.nf*3, self.nf*2, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(self.nf*2, self.nf, 3, 1, 1, bias=True)
        self.conv3_f1 = nn.Conv2d(self.nf, 2, 3, 1, 1, bias=True)
        self.conv3_f2 = nn.Conv2d(self.nf, 2, 3, 1, 1, bias=True )

    def forward(self, f1, f2, fe):
        fea = torch.cat([f1, f2, fe], dim=1)
        fea = self.conv2(self.lrelu(self.conv1(fea)))
        delta1 = self.conv3_f1(fea).permute(0,2,3,1)
        delta2 = self.conv3_f2(fea).permute(0,2,3,1)

        f1_modulated = nn.functional.grid_sample(f1, delta1)
        f2_modulated = nn.functional.grid_sample(f2, delta2)
        out = f1_modulated + f2_modulated
        return out


###################################################################################################
class SEBlock(nn.Module):
    def __init__(self, input_dim, reduction=16):
        super(SEBlock, self).__init__()
        mid = int(input_dim / reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, reduction),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
###################################################################################################


###################################################################################################
class SFTLayer(nn.Module):
    def __init__(self, nf):
        super(SFTLayer, self).__init__()
        self.nf = nf
        self.SFT_scale_conv0 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.SFT_scale_conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.SFT_shift_conv0 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)
        self.SFT_shift_conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1, bias=True)

    def forward(self, frame, event):
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(event), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(event), 0.1, inplace=True))
        return frame * scale + shift

class ResBlock_SFT(nn.Module):
    def __init__(self, nf):
        self.nf = nf
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(self.nf)
        self.conv0 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)
        self.sft1 = SFTLayer(self.nf)
        self.conv1 = nn.Conv2d(self.nf, self.nf, 3, 1, 1)

    def forward(self, frame, event):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(frame, event)
        fea = F.leaky_relu(fea, 0.1, inplace=True)
        fea = self.conv0(fea)
        fea = F.leaky_relu(self.sft1(fea, event), 0.1, inplace=True)
        fea = self.conv1(fea)
        return frame + fea # return a tuple containing features and conditions
###################################################################################################


if __name__ == '__main__':
    import argparse
    f1 = torch.rand((2, 64, 128, 128)).cuda()
    f2 = torch.rand((2, 64, 128, 128)).cuda()
    fe = torch.rand((2, 64, 128, 128)).cuda()
    
    model = EventModulation_AlignSeg(64).cuda()
    out = model(f1,f2,fe)
    print(out.shape)
