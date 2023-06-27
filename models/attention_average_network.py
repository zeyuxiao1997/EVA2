#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention_average_network.py
@Time    :   2021/07/28 10:26:51
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   timelens的第四阶段
'''
# here put the import lib


# import torch as th
# import torch.nn.functional as F
# from timelens import refine_warp_network, warp_network
# from timelens.superslomo import unet
from torch import nn
import torch
import torch as th
import torch.nn.functional as F
import models.refine_warp_network as refine_warp_network
import models.warp_network as warp_network

import models.superslomo as unet


def _pack_input_for_attention_computation(example):
    fusion = example["middle"]["fusion"]
    number_of_examples, _, height, width = fusion.size()
    return th.cat(
        [
            example["after"]["flow"],
            example["middle"]["after_refined_warped"],
            example["before"]["flow"],
            example["middle"]["before_refined_warped"],
            example["middle"]["fusion"],
            th.Tensor(example["middle"]["weight"])
            .view(-1, 1, 1, 1)
            .expand(number_of_examples, 1, height, width)
            .type(fusion.type()),
        ],
        dim=1,
    )


def _compute_weighted_average(attention, before_refined, after_refined, fusion):
    return (
        attention[:, 0, ...].unsqueeze(1) * before_refined
        + attention[:, 1, ...].unsqueeze(1) * after_refined
        + attention[:, 2, ...].unsqueeze(1) * fusion
    )


class AttentionAverage(refine_warp_network.RefineWarp):
    def __init__(self):
        warp_network.Warp.__init__(self)
        self.fusion_network = unet.UNet(2 * 3 + 2 * 5, 3, False)
        self.flow_refinement_network = unet.UNet(9, 4, False)
        self.attention_network = unet.UNet(14, 3, False)

    def run_fast(self, example):
        example['middle']['before_refined_warped'], \
        example['middle']['after_refined_warped'] = refine_warp_network.RefineWarp.run_fast(self, example)

        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example['middle']['before_refined_warped'],
            example['middle']['after_refined_warped'],
            example['middle']['fusion']
        )
        return average, attention

    def run_attention_averaging(self, example):
        refine_warp_network.RefineWarp.run_and_pack_to_example(self, example)
        attention_scores = self.attention_network(
            _pack_input_for_attention_computation(example)
        )
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            example["middle"]["before_refined_warped"],
            example["middle"]["after_refined_warped"],
            example["middle"]["fusion"],
        )
        return average, attention

    def run_and_pack_to_example(self, example):
        (
            example["middle"]["attention_average"],
            example["middle"]["attention"],
        ) = self.run_attention_averaging(example)

    def forward(self, example):
        return self.run_attention_averaging(example)



class AttentionAverageModify(nn.Module):
    def __init__(self):
        super(AttentionAverageModify, self).__init__()
        self.attention_network = unet.UNet(13, 3, False)
        
    def forward(self, before_refined, after_refined, before_flow, after_flow, stage1_output):
        number_of_examples, _, height, width = stage1_output.size()
        input = torch.cat([before_refined, after_refined, before_flow, after_flow, stage1_output], dim=1)
        attention_scores = self.attention_network(input)
        attention = F.softmax(attention_scores, dim=1)
        average = _compute_weighted_average(
            attention,
            before_refined, after_refined,stage1_output
        )

        return average
        
class AttentionAverageModify_Test(nn.Module):
    def __init__(self):
        super(AttentionAverageModify_Test, self).__init__()
        self.attention_network = unet.UNet(14, 3, False)
        
    def forward(self, input):
        # number_of_examples, _, height, width = stage1_output.size()
        attention_scores = self.attention_network(input)
        attention = F.softmax(attention_scores, dim=1)
        # average = _compute_weighted_average(
        #     attention,
        #     before_refined, after_refined,stage1_output
        # )

        # return average
        