#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vgg.py
@Time    :   2021/07/02 15:06:56
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib


from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(pretrained=False)
        vgg.load_state_dict(torch.load('/ghome/xiaozy/EventVideoFrameInterpolation/rpg_timelens/myCode/vgg19-dcbb9e9d.pth'))
        vgg_pretrained_features = vgg.features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return h_relu1


class VGGLoss(nn.Module):
    def __init__(self, gpu_id=0):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda(gpu_id)
        self.criterion = nn.MSELoss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # loss = 0
        # for i in range(len(x_vgg)):
        loss =  self.criterion(x_vgg, y_vgg.detach())
        return loss



if __name__ == "__main__":
    images = Variable(torch.ones(1, 3, 128, 128)).cuda()
    vgg = Vgg19()
    vgg.cuda()
    print("do forward...")
    outputs = vgg(images)
    print (outputs.size())   # (10, 100)
    print(torch.max(outputs))
