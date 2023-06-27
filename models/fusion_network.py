#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   fusion_network.py
@Time    :   2021/06/28 15:14:37
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   好像就是所谓的interpolation by synthesis module.
             直接嵌入网络就行
             这个网络是第一部分，首先要train这个部分
             相比于timeLens，这里把代码的一部分内容改了一下
'''
# here put the import lib


import torch
import models.superslomo as unet
from torch import nn

# def _pack(example):
#     return th.cat([example['before']['voxel_grid'],
#                    example['before']['rgb_image_tensor'],
#                    example['after']['voxel_grid'],
#                    example['after']['rgb_image_tensor']], dim=1)


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False) 
        # 应该是timeBins选的是5
        
    def forward(self, frame1, frame2, event_0_t, event_t_1):
        input = torch.cat([event_0_t, frame1, event_t_1, frame2], dim=1)
        out = self.fusion_network(input)
        return out
        
    # def run_fusion(self, example):
    #     return self.fusion_network(_pack(example))
        
    # def from_legacy_checkpoint(self, checkpoint_filename):
    #     checkpoint = th.load(checkpoint_filename)
    #     self.load_state_dict(checkpoint["networks"])

    # def run_and_pack_to_example(self, example):
    #     example['middle']['fusion'] = self.run_fusion(example)
        
    # def forward(self, example):
    #     return self.run_fusion(example)

class Fusion_Test(nn.Module):
    def __init__(self):
        super(Fusion_Test, self).__init__()
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False) 
        # 应该是timeBins选的是5
        
    def forward(self, input):
        # input = torch.cat([event_0_t, frame1, event_t_1, frame2], dim=1)
        out = self.fusion_network(input)
        return out
        