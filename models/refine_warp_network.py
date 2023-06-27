#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   refine_warp_network.py
@Time    :   2021/07/14 15:50:31
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   stage3的网络，warping refine的网路
'''
# here put the import lib


# import torch as th
# from timelens.common import warp
# from timelens import fusion_network, warp_network
# from timelens.superslomo import unet

from torch import nn
import torch
import models.warp as warp
import models.fusion_network as fusion_network 
import models.warp_network as warp_network
import models.superslomo as unet


def _pack_for_residual_flow_computation(example):
    tensors = [
        example["middle"]["{}_warped".format(packet)] for packet in ["after", "before"]
    ]
    tensors.append(example["middle"]["fusion"])
    return th.cat(tensors, dim=1)


def _pack_images_for_second_warping(example):
    return th.cat(
        [example["middle"]["after_warped"], example["middle"]["before_warped"]],
    )


def _pack_output_to_example(example, output):
    (
        example["middle"]["before_refined_warped"],
        example["middle"]["after_refined_warped"],
        example["middle"]["before_refined_warped_invalid"],
        example["middle"]["after_refined_warped_invalid"],
        example["before"]["residual_flow"],
        example["after"]["residual_flow"],
    ) = output


class RefineWarp(warp_network.Warp, fusion_network.Fusion):
    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)

    def run_refine_warp(self, example):
        warp_network.Warp.run_and_pack_to_example(self, example)
        fusion_network.Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = th.chunk(residual, 2, dim=1)
        residual = th.cat([after_residual, before_residual], dim=0)
        refined, refined_invalid = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = th.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = th.chunk(
            refined_invalid.detach(), 2)
        return (
            before_refined,
            after_refined,
            before_refined_invalid,
            after_refined_invalid,
            before_residual,
            after_residual,
        )


    def run_and_pack_to_example(self, example):
        _pack_output_to_example(example, self.run_refine_warp(example))

    def forward(self, before_warped, after_warped, before_flow, after_flow, before_warped_invalid, after_warped_invalid):
        return self.run_refine_warp(example)



class RefineWarpModify(nn.Module):
    def __init__(self):
        super(RefineWarpModify, self).__init__()
        self.flow_refinement_network = unet.UNet(9, 4, False)

    def run_refine_warp(self, example):
        warp_network.Warp.run_and_pack_to_example(self, example)
        fusion_network.Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([after_residual, before_residual], dim=0)

        
        refined, refined_invalid = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = th.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = th.chunk(
            refined_invalid.detach(), 2)
        return (
            before_refined,
            after_refined,
            before_refined_invalid,
            after_refined_invalid,
            before_residual,
            after_residual,
        )


    def forward(self, stage2_before_warped, stage2_after_warped, stage1_output):
        input = torch.cat([stage2_after_warped, stage2_before_warped, stage1_output],dim=1)
        residual = self.flow_refinement_network(input)

        (after_residual, before_residual) = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([after_residual, before_residual], dim=0)

        warped_input = torch.cat([stage2_after_warped, stage2_before_warped])

        refined, refined_invalid = warp.backwarp_2d(
            source=warped_input,
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = torch.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = torch.chunk(
            refined_invalid.detach(), 2)


        return before_refined, after_refined, before_refined_invalid, after_refined_invalid, before_residual, after_residual


class RefineWarpModify_Test(nn.Module):
    def __init__(self):
        super(RefineWarpModify_Test, self).__init__()
        self.flow_refinement_network = unet.UNet(9, 4, False)

    def run_refine_warp(self, example):
        warp_network.Warp.run_and_pack_to_example(self, example)
        fusion_network.Fusion.run_and_pack_to_example(self, example)
        residual = self.flow_refinement_network(
            _pack_for_residual_flow_computation(example)
        )
        (after_residual, before_residual) = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([after_residual, before_residual], dim=0)

        
        refined, refined_invalid = warp.backwarp_2d(
            source=_pack_images_for_second_warping(example),
            y_displacement=residual[:, 0, ...],
            x_displacement=residual[:, 1, ...],
        )
        
        (after_refined, before_refined) = th.chunk(refined, 2)
        (after_refined_invalid, before_refined_invalid) = th.chunk(
            refined_invalid.detach(), 2)
        return (
            before_refined,
            after_refined,
            before_refined_invalid,
            after_refined_invalid,
            before_residual,
            after_residual,
        )


    def forward(self, input):
        residual = self.flow_refinement_network(input)

        (after_residual, before_residual) = torch.chunk(residual, 2, dim=1)
        residual = torch.cat([after_residual, before_residual], dim=0)

        warped_input = torch.rand(1,2,448,256)

        # refined, refined_invalid = warp.backwarp_2d(
        #     source=warped_input,
        #     y_displacement=residual[:, 0, ...],
        #     x_displacement=residual[:, 1, ...],
        # )
        
        # (after_refined, before_refined) = torch.chunk(refined, 2)
        # (after_refined_invalid, before_refined_invalid) = torch.chunk(
        #     refined_invalid.detach(), 2)


        # return before_refined, after_refined, before_refined_invalid, after_refined_invalid, before_residual, after_residual
