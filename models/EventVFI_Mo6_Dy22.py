#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   EventVFI_Mo5_Dy22.py
@Time    :   2021/07/08 20:21:23
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

# import functools
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import arch_util as arch_util
# import tmpModules1 as tmpModules1
# import tmpModules2 as tmpModules2
# try:
#     from dcn.deform_conv import ModulatedDeformConvPack as DCN
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')


import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.arch_util as arch_util
import models.tmpModules1 as tmpModules1
import models.tmpModules2 as tmpModules2
try:
    from models.dcn.deform_conv import ModulatedDeformConvPack as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')



class EventVFI1(nn.Module):
    """
    网络的模块细节见file文件处；
    这里encoder和decoder的feature连接为concat，用concat似乎更好
    """
    def __init__(self):
        super(EventVFI1, self).__init__()

        ks = 3 # kernel size        
        # self.nf = arg.nf
        ########## multi-scale feature extractor for RGB & event ##########
        self.conv_rgb_1_1 = arch_util.conv(3, 32, kernel_size=ks, stride=1)
        self.conv_rgb_1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.conv_rgb_1_3 = arch_util.resnet_block(32, kernel_size=ks)

        self.conv_rgb_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_rgb_2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.conv_rgb_2_3 = arch_util.resnet_block(64, kernel_size=ks)

        self.conv_rgb_3_1 = arch_util.conv(64, 96, kernel_size=ks, stride=2)
        self.conv_rgb_3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.conv_rgb_3_3 = arch_util.resnet_block(96, kernel_size=ks)

        self.conv_event_1_1 = arch_util.conv(5, 32, kernel_size=ks, stride=1)
        self.conv_event_1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.conv_event_1_3 = arch_util.resnet_block(32, kernel_size=ks)

        self.conv_event_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_event_2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.conv_event_2_3 = arch_util.resnet_block(64, kernel_size=ks)

        self.conv_event_3_1 = arch_util.conv(64, 96, kernel_size=ks, stride=2)
        self.conv_event_3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.conv_event_3_3 = arch_util.resnet_block(96, kernel_size=ks)

        ######################## EventModulation #############################
        self.EventModulationL1_0_t = tmpModules1.EventModulation6(32)
        self.EventModulationL1_t_1 = tmpModules1.EventModulation6(32)
        self.EventModulationL2_0_t = tmpModules1.EventModulation6(64)
        self.EventModulationL2_t_1 = tmpModules1.EventModulation6(64)
        self.EventModulationL3_0_t = tmpModules1.EventModulation6(96)
        self.EventModulationL3_t_1 = tmpModules1.EventModulation6(96)

        ######################## Event+KPN #############################
        self.DynamicEventL3 = tmpModules2.DynamicEvent22(96, 96, 96)
        self.DynamicEventL2 = tmpModules2.DynamicEvent22(64, 64, 64)
        self.DynamicEventL1 = tmpModules2.DynamicEvent22(32, 32, 32)


        ############################# Decoder #############################
        self.upconv3_i = arch_util.conv(96, 96, kernel_size=ks, stride=1)
        self.upconv3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.upconv3_1 = arch_util.resnet_block(96, kernel_size=ks)

        self.upconv2_u = arch_util.upconv(96, 64)
        self.upconv2_i = arch_util.conv(128, 64, kernel_size=ks,stride=1)
        self.upconv2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.upconv2_1 = arch_util.resnet_block(64, kernel_size=ks)

        self.upconv1_u = arch_util.upconv(64, 32)
        self.upconv1_i = arch_util.conv(64, 32, kernel_size=ks,stride=1)
        self.upconv1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.upconv1_1 = arch_util.resnet_block(32, kernel_size=ks)

        self.img_prd = arch_util.conv(32, 3, kernel_size=ks, stride=1)


    def forward(self, frame1, frame2, event_0_t, event_t_1):
        # encoder
        # sharing weights
        frame1_L2 = F.interpolate(frame1, scale_factor=0.5, mode="bilinear")
        frame1_L3 = F.interpolate(frame1_L2, scale_factor=0.5, mode="bilinear")

        frame2_L2 = F.interpolate(frame2, scale_factor=0.5, mode="bilinear")
        frame2_L3 = F.interpolate(frame2_L2, scale_factor=0.5, mode="bilinear")

        conv1_frame1 = self.conv_rgb_1_3(self.conv_rgb_1_2(self.conv_rgb_1_1(frame1)))
        conv2_frame1 = self.conv_rgb_2_3(self.conv_rgb_2_2(self.conv_rgb_2_1(conv1_frame1)))
        conv3_frame1 = self.conv_rgb_3_3(self.conv_rgb_3_2(self.conv_rgb_3_1(conv2_frame1)))

        conv1_frame2 = self.conv_rgb_1_3(self.conv_rgb_1_2(self.conv_rgb_1_1(frame2)))
        conv2_frame2 = self.conv_rgb_2_3(self.conv_rgb_2_2(self.conv_rgb_2_1(conv1_frame2)))
        conv3_frame2 = self.conv_rgb_3_3(self.conv_rgb_3_2(self.conv_rgb_3_1(conv2_frame2)))

        conv1_event_0_t = self.conv_event_1_3(self.conv_event_1_2(self.conv_event_1_1(event_0_t)))
        conv2_event_0_t = self.conv_event_2_3(self.conv_event_2_2(self.conv_event_2_1(conv1_event_0_t)))
        conv3_event_0_t = self.conv_event_3_3(self.conv_event_3_2(self.conv_event_3_1(conv2_event_0_t)))

        conv1_event_t_1 = self.conv_event_1_3(self.conv_event_1_2(self.conv_event_1_1(event_t_1)))
        conv2_event_t_1 = self.conv_event_2_3(self.conv_event_2_2(self.conv_event_2_1(conv1_event_t_1)))
        conv3_event_t_1 = self.conv_event_3_3(self.conv_event_3_2(self.conv_event_3_1(conv2_event_t_1)))


        FrameL3_0_t = self.EventModulationL3_0_t(conv3_frame1, frame1_L3, conv3_event_0_t)   # torch.Size([2, 96, 64, 64])
        FrameL3_t_1 = self.EventModulationL3_t_1(conv3_frame2, frame2_L3, conv3_event_t_1)   # torch.Size([2, 96, 64, 64])
        FrameL3 = self.DynamicEventL3(FrameL3_0_t, FrameL3_t_1, conv3_event_0_t, conv3_event_t_1)   # torch.Size([2, 96, 64, 64])

        FrameL2_0_t = self.EventModulationL2_0_t(conv2_frame1, frame1_L2, conv2_event_0_t)   # torch.Size([2, 64, 128, 128])
        FrameL2_t_1 = self.EventModulationL2_t_1(conv2_frame2, frame2_L2, conv2_event_t_1)   # torch.Size([2, 64, 128, 128])
        FrameL2 = self.DynamicEventL2(FrameL2_0_t, FrameL2_t_1, conv2_event_0_t, conv2_event_t_1)   # torch.Size([2, 64, 128, 128])

        FrameL1_0_t = self.EventModulationL1_0_t(conv1_frame1, frame1, conv1_event_0_t)   # torch.Size([2, 32, 256, 256])
        FrameL1_t_1 = self.EventModulationL1_t_1(conv1_frame2, frame2, conv1_event_t_1)   # torch.Size([2, 32, 256, 256])
        FrameL1 = self.DynamicEventL1(FrameL1_0_t, FrameL1_t_1, conv1_event_0_t, conv1_event_t_1)   # torch.Size([2, 32, 256, 256])
        
        cat3 = self.upconv3_i(FrameL3) # torch.Size([2, 64, 128, 128])
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(cat3)))  # torch.Size([2, 64, 128, 128])
        cat2 = self.upconv2_i(torch.cat((upconv2,FrameL2),1))   # torch.Size([2, 64, 128, 128])
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(cat2)))  # torch.Size([2, 32, 256, 256])
        cat1 = self.upconv1_i(torch.cat((upconv1,FrameL1),1))   # torch.Size([2, 32, 256, 256])
        img_prd = self.img_prd(self.upconv1_1(self.upconv1_2(cat1)))    # torch.Size([2, 3, 256, 256])

        return img_prd 





class EventVFI2(nn.Module):
    """
    网络的模块细节见file文件处；
    这里encoder和decoder的feature连接为相加
    """
    def __init__(self):
        super(EventVFI2, self).__init__()

        ks = 3 # kernel size        
        # self.nf = arg.nf
        ########## multi-scale feature extractor for RGB & event ##########
        self.conv_rgb_1_1 = arch_util.conv(3, 32, kernel_size=ks, stride=1)
        self.conv_rgb_1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.conv_rgb_1_3 = arch_util.resnet_block(32, kernel_size=ks)

        self.conv_rgb_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_rgb_2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.conv_rgb_2_3 = arch_util.resnet_block(64, kernel_size=ks)

        self.conv_rgb_3_1 = arch_util.conv(64, 96, kernel_size=ks, stride=2)
        self.conv_rgb_3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.conv_rgb_3_3 = arch_util.resnet_block(96, kernel_size=ks)

        self.conv_event_1_1 = arch_util.conv(5, 32, kernel_size=ks, stride=1)
        self.conv_event_1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.conv_event_1_3 = arch_util.resnet_block(32, kernel_size=ks)

        self.conv_event_2_1 = arch_util.conv(32, 64, kernel_size=ks, stride=2)
        self.conv_event_2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.conv_event_2_3 = arch_util.resnet_block(64, kernel_size=ks)

        self.conv_event_3_1 = arch_util.conv(64, 96, kernel_size=ks, stride=2)
        self.conv_event_3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.conv_event_3_3 = arch_util.resnet_block(96, kernel_size=ks)

        ######################## EventModulation #############################
        self.EventModulationL1_0_t = tmpModules1.EventModulation6(32)
        self.EventModulationL1_t_1 = tmpModules1.EventModulation6(32)
        self.EventModulationL2_0_t = tmpModules1.EventModulation6(64)
        self.EventModulationL2_t_1 = tmpModules1.EventModulation6(64)
        self.EventModulationL3_0_t = tmpModules1.EventModulation6(96)
        self.EventModulationL3_t_1 = tmpModules1.EventModulation6(96)

        ######################## Event+KPN #############################
        self.DynamicEventL3 = tmpModules2.DynamicEvent22(96, 96, 96)
        self.DynamicEventL2 = tmpModules2.DynamicEvent22(64, 64, 64)
        self.DynamicEventL1 = tmpModules2.DynamicEvent22(32, 32, 32)


        ############################# Decoder #############################
        self.upconv3_i = arch_util.conv(96, 96, kernel_size=ks, stride=1)
        self.upconv3_2 = arch_util.resnet_block(96, kernel_size=ks)
        self.upconv3_1 = arch_util.resnet_block(96, kernel_size=ks)

        self.upconv2_u = arch_util.upconv(96, 64)
        self.upconv2_i = arch_util.conv(64, 64, kernel_size=ks,stride=1)
        self.upconv2_2 = arch_util.resnet_block(64, kernel_size=ks)
        self.upconv2_1 = arch_util.resnet_block(64, kernel_size=ks)

        self.upconv1_u = arch_util.upconv(64, 32)
        self.upconv1_i = arch_util.conv(32, 32, kernel_size=ks,stride=1)
        self.upconv1_2 = arch_util.resnet_block(32, kernel_size=ks)
        self.upconv1_1 = arch_util.resnet_block(32, kernel_size=ks)

        self.img_prd = arch_util.conv(32, 3, kernel_size=ks, stride=1)


    def forward(self, frame1, frame2, event_0_t, event_t_1):
        # encoder
        # sharing weights

        frame1_L2 = F.interpolate(frame1, scale_factor=0.5, mode="bilinear")
        frame1_L3 = F.interpolate(frame1_L2, scale_factor=0.5, mode="bilinear")

        frame2_L2 = F.interpolate(frame2, scale_factor=0.5, mode="bilinear")
        frame2_L3 = F.interpolate(frame2_L2, scale_factor=0.5, mode="bilinear")

        conv1_frame1 = self.conv_rgb_1_3(self.conv_rgb_1_2(self.conv_rgb_1_1(frame1)))
        conv2_frame1 = self.conv_rgb_2_3(self.conv_rgb_2_2(self.conv_rgb_2_1(conv1_frame1)))
        conv3_frame1 = self.conv_rgb_3_3(self.conv_rgb_3_2(self.conv_rgb_3_1(conv2_frame1)))

        conv1_frame2 = self.conv_rgb_1_3(self.conv_rgb_1_2(self.conv_rgb_1_1(frame2)))
        conv2_frame2 = self.conv_rgb_2_3(self.conv_rgb_2_2(self.conv_rgb_2_1(conv1_frame2)))
        conv3_frame2 = self.conv_rgb_3_3(self.conv_rgb_3_2(self.conv_rgb_3_1(conv2_frame2)))

        conv1_event_0_t = self.conv_event_1_3(self.conv_event_1_2(self.conv_event_1_1(event_0_t)))
        conv2_event_0_t = self.conv_event_2_3(self.conv_event_2_2(self.conv_event_2_1(conv1_event_0_t)))
        conv3_event_0_t = self.conv_event_3_3(self.conv_event_3_2(self.conv_event_3_1(conv2_event_0_t)))

        conv1_event_t_1 = self.conv_event_1_3(self.conv_event_1_2(self.conv_event_1_1(event_t_1)))
        conv2_event_t_1 = self.conv_event_2_3(self.conv_event_2_2(self.conv_event_2_1(conv1_event_t_1)))
        conv3_event_t_1 = self.conv_event_3_3(self.conv_event_3_2(self.conv_event_3_1(conv2_event_t_1)))


        FrameL3_0_t = self.EventModulationL3_0_t(conv3_frame1, frame1_L3, conv3_event_0_t)   # torch.Size([2, 96, 64, 64])
        FrameL3_t_1 = self.EventModulationL3_t_1(conv3_frame2, frame2_L3, conv3_event_t_1)   # torch.Size([2, 96, 64, 64])
        FrameL3 = self.DynamicEventL3(FrameL3_0_t, FrameL3_t_1, conv3_event_0_t, conv3_event_t_1)   # torch.Size([2, 96, 64, 64])

        FrameL2_0_t = self.EventModulationL2_0_t(conv2_frame1, frame1_L2, conv2_event_0_t)   # torch.Size([2, 64, 128, 128])
        FrameL2_t_1 = self.EventModulationL2_t_1(conv2_frame2, frame2_L2, conv2_event_t_1)   # torch.Size([2, 64, 128, 128])
        FrameL2 = self.DynamicEventL2(FrameL2_0_t, FrameL2_t_1, conv2_event_0_t, conv2_event_t_1)   # torch.Size([2, 64, 128, 128])

        FrameL1_0_t = self.EventModulationL1_0_t(conv1_frame1, frame1, conv1_event_0_t)   # torch.Size([2, 32, 256, 256])
        FrameL1_t_1 = self.EventModulationL1_t_1(conv1_frame2, frame2, conv1_event_t_1)   # torch.Size([2, 32, 256, 256])
        FrameL1 = self.DynamicEventL1(FrameL1_0_t, FrameL1_t_1, conv1_event_0_t, conv1_event_t_1)   # torch.Size([2, 32, 256, 256])
        
        cat3 = self.upconv3_i(FrameL3) # torch.Size([2, 64, 128, 128])
        upconv2 = self.upconv2_u(self.upconv3_1(self.upconv3_2(cat3)))  # torch.Size([2, 64, 128, 128])
        cat2 = self.upconv2_i(upconv2+FrameL2)  # torch.Size([2, 64, 128, 128])
        upconv1 = self.upconv1_u(self.upconv2_1(self.upconv2_2(cat2)))  # torch.Size([2, 32, 256, 256])
        cat1 = self.upconv1_i(upconv1+FrameL1)   # torch.Size([2, 32, 256, 256])
        img_prd = self.img_prd(self.upconv1_1(self.upconv1_2(cat1)))    # torch.Size([2, 3, 256, 256])

        return img_prd




if __name__ == "__main__":
    input = torch.rand(2,3,256,256).cuda()
    model = EventVFI1().cuda()
    output = model(input,input,input,input)
    print(output.shape)
