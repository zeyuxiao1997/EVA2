#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   warp_network.py
@Time    :   2021/06/28 17:33:28
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   应该是2阶段的网络
             从timeLens魔改来的
'''
# here put the import lib

import torch as th
import torch
import models.superslomo as unet
from torch import nn
import models.warp as warp


def _pack_voxel_grid_for_flow_estimation(example):
    return th.cat(
        [example["before"]["reversed_voxel_grid"], example["after"]["voxel_grid"]]
    )


def _pack_images_for_warping(example):
    return th.cat(
        [example["before"]["rgb_image_tensor"], example["after"]["rgb_image_tensor"]]
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_warped"],
        example["middle"]["after_warped"],
        example["before"]["flow"],
        example["after"]["flow"],
        example["middle"]["before_warped_invalid"],
        example["middle"]["after_warped_invalid"],
    ) = output


class WarpingBasedInterpolationModule(nn.Module):
    def __init__(self):
        super(WarpingBasedInterpolationModule, self).__init__()
        self.flow_network = unet.UNet(5, 2, False)

    def forward(self, frame1, frame2, event_0_t_reverse, event_t_1):
        voxel_grid_for_flow_estimation = torch.cat([event_0_t_reverse, event_t_1])
        images_for_warping = torch.cat([frame1, frame2])
        flow = self.flow_network(voxel_grid_for_flow_estimation)
        warped, warped_invalid = warp.backwarp_2d(
            source=images_for_warping,
            y_displacement=flow[:, 0, ...],
            x_displacement=flow[:, 1, ...],
        )
        (before_flow, after_flow) = th.chunk(flow, chunks=2)
        (before_warped, after_warped) = th.chunk(warped, chunks=2)
        (before_warped_invalid, after_warped_invalid) = th.chunk(
            warped_invalid.detach(), chunks=2
        )
        return before_warped, after_warped, before_flow, after_flow, before_warped_invalid, after_warped_invalid


class Warp(nn.Module):
    def __init__(self):
        super(Warp, self).__init__()
        self.flow_network = unet.UNet(5, 2, False)

    def from_legacy_checkpoint(self, checkpoint_filename):
        checkpoint = th.load(checkpoint_filename)
        self.load_state_dict(checkpoint["networks"])

    def run_warp(self, example):
        flow = self.flow_network(_pack_voxel_grid_for_flow_estimation(example))
        warped, warped_invalid = warp.backwarp_2d(
            source=_pack_images_for_warping(example),
            y_displacement=flow[:, 0, ...],
            x_displacement=flow[:, 1, ...],
        )
        (before_flow, after_flow) = th.chunk(flow, chunks=2)
        (before_warped, after_warped) = th.chunk(warped, chunks=2)
        (before_warped_invalid, after_warped_invalid) = th.chunk(
            warped_invalid.detach(), chunks=2
        )
        return (
            before_warped,
            after_warped,
            before_flow,
            after_flow,
            before_warped_invalid,
            after_warped_invalid,
        )
    
    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_warp(example))

    def forward(self, example):
        return self.run_warp(example)


class WarpingBasedInterpolationModule_Test(nn.Module):
    def __init__(self):
        super(WarpingBasedInterpolationModule_Test, self).__init__()
        self.flow_network = unet.UNet(5, 2, False)

    def forward(self, voxel_grid_for_flow_estimation):
        images_for_warping = torch.rand(1,2,448,256)
        voxel_grid_for_flow_estimation = torch.rand(1,5,448,256)
        flow = self.flow_network(voxel_grid_for_flow_estimation)
        warped, warped_invalid = warp.backwarp_2d(
            source=images_for_warping,
            y_displacement=flow[:, 0, ...],
            x_displacement=flow[:, 1, ...],
        )
        # (before_flow, after_flow) = th.chunk(flow, chunks=2)
        # (before_warped, after_warped) = th.chunk(warped, chunks=2)
        # (before_warped_invalid, after_warped_invalid) = th.chunk(
        #     warped_invalid.detach(), chunks=2
        # )
        # return before_warped, after_warped, before_flow, after_flow, before_warped_invalid, after_warped_invalid