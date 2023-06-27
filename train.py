#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train_Mo2_Dy2_M1.py
@Time    :   2021/07/08 12:40:53
@Author  :   Zeyu Xiao
@Version :   1.0
@Contact :   zeyuxiao@mail.ustc.edu.cn, zeyuxiao1997@163.com
@License :   (C)Copyright 2019-2024
@Desc    :   
'''
# here put the import lib

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
from models import Losses
from models import EventVFI_Mo6_Dy21_HalfChannel
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
from dataloader.dataset import *
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import random
import numpy as np
from tensorboardX import *
import torchvision.utils as visionutils
import math
# from thop import profile


def train():
    opt.train_batch_size = 4
    opt.num_workers = 0
    opt.ModelName = 'Mo6_Dy22_1_half'
    file4train = '/ghome/xiaozy/EventVideoFrameInterpolation/evfi/dataloader/train.yml'
    file4test = '/ghome/xiaozy/EventVideoFrameInterpolation/evfi/dataloader/test.yml'
    # opt.model_path = '/gdata2/xiaozy/Result/Mo2_Dy2_1/DeblurSR_iter30000+PSNR496127.65713536260611_02_12_57.pth'

    print(opt)
    Best = 0
    transform = transforms.Compose([transforms.ToTensor()])
    opt.manualSeed = random.randint(1, 10000)
    opt.saveDir = os.path.join(opt.exp, opt.ModelName)
    create_exp_dir(opt.saveDir)
    device = torch.device("cuda:7")

    # train_data = DatasetFromFolder(opt)
    with open(file4train) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
    
    dataset4train = H5Dataset(config)
    train_dataloader = DataLoader(dataset4train,
                        batch_size=opt.train_batch_size,
                        shuffle=True,
                        num_workers=opt.num_workers,
                        drop_last=True)
    print('length of train_dataloader: ',len(train_dataloader)) # 6000


    with open(file4test) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
    
    dataset4test = H5Dataset(config)
    test_dataloader = DataLoader(dataset4test,
                        batch_size=1,
                        shuffle=False,
                        num_workers=opt.num_workers,
                        drop_last=False)
    print('length of train_dataloader: ',len(test_dataloader)) # 6000

    # print(train_dataloader[0])

    last_epoch = 0

    ## initialize loss writer and logger
    ##############################################################
    loss_dir = os.path.join(opt.saveDir, 'loss')
    loss_writer = SummaryWriter(loss_dir)
    print("loss dir", loss_dir)
    trainLogger = open('%s/train.log' % opt.saveDir, 'w')
    ##############################################################

    model = EventVFI_Mo6_Dy21_HalfChannel.EventVFI1()
    model.train()
    model.cuda()

    criterionCharb = Losses.CharbonnierLoss()
    criterionCharb.cuda()


    lr = opt.lr
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=lr,
        betas=(opt.beta1, opt.beta2)
    )

    iteration = 0

    if opt.model_path != '':
        map_location = lambda storage, loc: storage
        checkpoint = torch.load(opt.model_path, map_location=map_location)
        last_epoch = checkpoint["epoch"]
        optimizer_state = checkpoint["optimizer"]
        optimizer.load_state_dict(optimizer_state)
        iteration = checkpoint["iteration"]
        model.load_state_dict(checkpoint["model"])
        lr = checkpoint["lr"]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('load pretrained')
    AllPSNR = 0
    for epoch in range(opt.max_epoch):
        if epoch < last_epoch:
            continue
        for _, batch in enumerate(train_dataloader, 0):
            iteration += 1 
            # print(batch)
            frame0 = batch['input']['frame0']
            frame1 = batch['input']['frame1']
            event0 = batch['input']['event0']
            event1 = batch['input']['event1']
            target = batch['gt']
            # print(iteration, name)
            # print(iteration, frame0.shape)
            frame0, frame1, event0, event1, target = frame0.float().cuda(), frame1.float().cuda(), event0.float().cuda(), event1.float().cuda(), target.float().cuda()
            frame0, frame1, event0, event1, target = frame0.cuda(), frame1.cuda(), event0.cuda(), event1.cuda(), target.float().cuda()
            # print(target.dtype)
            
            out = model(frame0, frame1, event0, event1)

            
            # print(out.shape)
            optimizer.zero_grad()
            
            CharbLoss1 = criterionCharb(out, target)
            AllLoss = CharbLoss1
            AllLoss.backward()
            optimizer.step()

            prediction = torch.clamp(out,0.0,1.0)

            if iteration%2 == 0:
                PPsnr = compute_psnr(tensor2np(prediction[0,:,:,:]),tensor2np(target[0,:,:,:]))
                if PPsnr==float('inf'):
                    PPsnr=50
                AllPSNR += PPsnr
                print('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss1.item(), PPsnr))
                trainLogger.write(('[%d/%d][%d] AllLoss:%.10f|CharbLoss:%.10f|PSNR:%.6f'
                    % (epoch, opt.max_epoch, iteration,
                    AllLoss.item(),CharbLoss1.item(), PPsnr))+'\n')

                loss_writer.add_scalar('CharbLoss4train', CharbLoss1.item(), iteration)
                loss_writer.add_scalar('PSNR4train', PPsnr, iteration)
                trainLogger.flush()

            if iteration%2000 == 0:
                loss_writer.add_image('Prediction', prediction[0,:,:,:], iteration) # x.size= (3, 266, 530) (C*H*W)
                loss_writer.add_image('target', target[0,:,:,:], iteration)
                loss_writer.add_image('input1', frame0[0,:,:,:], iteration)
                loss_writer.add_image('input2', frame1[0,:,:,:], iteration)

                
            # if iteration % opt.saveStep == 0:
            #     for _, (batch,name) in enumerate(train_dataloader, 0):
            #         iteration += 1 

            #         frame0 = batch['input']['frame0']
            #         frame1 = batch['input']['frame1']
            #         event0 = batch['input']['event0']
            #         event1 = batch['input']['event1']
            #         target = batch['gt']
            #         # print(iteration, name)
            #         # print(iteration, frame0.shape)
            #         frame0, frame1, event0, event1, target = frame0.float().cuda(), frame1.float().cuda(), event0.float().cuda(), event1.float().cuda(), target.float().cuda()
            #         frame0, frame1, event0, event1, target = frame0.cuda(), frame1.cuda(), event0.cuda(), event1.cuda(), target.float().cuda()
            #         # print(target.dtype)
                    
            #         out = model(frame0, frame1, event0, event1)
            #         print(out.shape)

            #     prefix = opt.saveDir+'/EventVFI_iter{}'.format(iteration)+'+PSNR'+str(Best)
            #     file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
            #     checkpoint = {
            #         'epoch': epoch,
            #         'iteration': iteration,
            #         "optimizer": optimizer.state_dict(),
            #         "model": model.state_dict(),
            #         "lr": lr
            #     }
            #     torch.save(checkpoint, file_name)
            #     print('model saved to ==>'+file_name)
            #     AllPSNR = 0


            if iteration % opt.saveStep == 0:
                is_best = AllPSNR > Best
                Best = max(AllPSNR, Best)
                if is_best or iteration%(opt.saveStep*5)==0:
                    prefix = opt.saveDir+'/EVFI_iter{}'.format(iteration)+'+PSNR'+str(Best)
                    file_name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
                    checkpoint = {
                        'epoch': epoch,
                        'iteration': iteration,
                        "optimizer": optimizer.state_dict(),
                        "model": model.state_dict(),
                        "lr": lr
                    }
                torch.save(checkpoint, file_name)
                print('model saved to ==>'+file_name)
                AllPSNR = 0


            if (iteration + 1) % opt.decay_step == 0:
                lr = lr * opt.lr_decay
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    trainLogger.close()



if __name__ == "__main__":
    train()
