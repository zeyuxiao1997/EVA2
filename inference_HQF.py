
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
from models import EventVFI_Mo6_Dy21
from utils.myutils import *
import torch.backends.cudnn as cudnn
from options import *
# from dataloader4test.dataset import *
from torchvision.transforms import ToPILImage
import torchvision.transforms as transforms
import random
import numpy as np
from tensorboardX import *
import torchvision.utils as visionutils
import math

from numpy import mean
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import pandas as pd
from collections import defaultdict
# local modules
from myutils_weng.utils import *
from dataloader0823.h5dataset import *
from dataloader0823.h5dataloader import *
from loss import *
from dataloader0823.encodings import *
from myutils_weng.vis_events.matplotlib_plot_events import *
from skimage.metrics import structural_similarity
from skimage.measure import compare_ssim as ssim


# from thop import profile

# 48.27658004510247
# 0.07266846754637929


# def Compu_SSIM(pred, tgt):
#     # loss = structural_similarity(pred.squeeze().cpu().numpy(), tgt.squeeze().cpu().numpy(), multichannel=True)
#     return structural_similarity(pred.squeeze().cpu().numpy(), tgt.squeeze().cpu().numpy(), K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,multichannel=True)
#     # return loss

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.float().cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)


def Compu_SSIM(im1, im2):
    isRGB = len(im1.shape) == 3 and im1.shape[-1] == 3
    s = ssim(im1, im2, K1=0.01, K2=0.03, gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
             multichannel=isRGB)
    return s

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/gdata2/xiaozy/Result/Mo6_Dy22_1_finetune-MD/EVFI_iter292000+PSNR21139.6846144616320823_05_20_35.pth')
    parser.add_argument('--yaml_config', type=str, default='/ghome/xiaozy/EventVideoFrameInterpolation/evfi_inference_weng/config_weng/test_hqf_EVFINet.yml')
    parser.add_argument('--skip_frame', type=int, default=None) 
    parser.add_argument('--device', type=str, default='cuda:0')

    return parser.parse_args()

@torch.no_grad()
def main():
    # config['experiment']
    flags = get_flags()    

    model_path = flags.model_path
    yaml_config = flags.yaml_config
    skip_frame = flags.skip_frame
    device = torch.device(flags.device)

    with open(yaml_config) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)

    if skip_frame is not None:
        config['dataloader']['dataset'].update({'skip_frame': skip_frame})
    print(config)

# EVFI_iter290000+PSNR231587.16998099320725_10_19_02
# EVFI_iter210000+PSNR228574.460153580760725_13_07_39

    # modelsall = os.listdir(model_path)
    # for item in modelsall:
    #     if item[-3:]=='pth':
    #         model_path_all = os.path.join(model_path,item)
    #         print(model_path_all)
    #         namepth = item[:-4]
    #         print(namepth)

    output_path = config['output_path']
    root_path = os.path.join(output_path, config['experiment'])
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(root_path, exist_ok=True)

    # assert os.path.isfile(model_path)
    # model_cpt = torch.load(model_path, map_location='cpu')
    # print(f'Load model from: {model_path}...')

    vis = event_visualisation()
    logger = Logger_yaml(os.path.join(root_path, 'inference.yml'))
    logger.log_dict(config, 'dataset_config')

    # build metrics
    compute_lpips = perceptual_loss(net='alex')
    compute_ssim = ssim_loss()
    compute_psnr = psnr_loss()

    # build model
    # time_bins_cpt = model_cpt['config']['arch']['args']['unet_kwargs']['num_bins']
    # assert time_bins_cpt == config['dataloader']['dataset']['time_bins'], 'Time bins from checkpoint is different from the one of config!'
    
    
    
    # model = eval(model_cpt['config']['arch']['type'])(**model_cpt['config']['arch']['args'])
    model = EventVFI_Mo6_Dy21.EventVFI1()
    
# model.eval()



# if opt.model_path != '':
    map_location = lambda storage, loc: storage
    checkpoint = torch.load(model_path, map_location=map_location)
#     last_epoch = checkpoint["epoch"]
#     optimizer_state = checkpoint["optimizer"]
#     optimizer.load_state_dict(optimizer_state)
#     iteration = checkpoint["iteration"]
    model.load_state_dict(checkpoint["model"])
#     lr = checkpoint["lr"]
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
    print('load pretrained')


    # model.load_state_dict(model_cpt['state_dict'])
    model.to(device)
    model.eval()

    # build dataset
    dataloader = H5Dataloader(config['dataloader'])

    # metric_track = MetricTracker(['ssim', 'psnr', 'lpips'])
    metric_track = MetricTracker(['ssim', 'psnr', 'lpips', 'time'])

    metric_track.reset()
    # model.reset_states()
    for i, (inputs) in enumerate(tqdm(dataloader, total=len(dataloader))):

        step_dir = os.path.join(root_path, str(i))
        if not os.path.exists(step_dir):
            os.makedirs(step_dir)
        if not os.path.exists(os.path.join(step_dir, 'gt')):
            os.makedirs(os.path.join(step_dir, 'gt'))
        if not os.path.exists(os.path.join(step_dir, 'pred_frame')):
            os.makedirs(os.path.join(step_dir, 'pred_frame'))
        if not os.path.exists(os.path.join(step_dir, 'left_event')):
            os.makedirs(os.path.join(step_dir, 'left_event'))
        if not os.path.exists(os.path.join(step_dir, 'right_event')):
            os.makedirs(os.path.join(step_dir, 'right_event'))

        # os.makedirs(step_dir, exist_ok=False)
        # os.makedirs(os.path.join(step_dir, 'gt'), exist_ok=False)
        # os.makedirs(os.path.join(step_dir, 'pred_frame'), exist_ok=False)
        # os.makedirs(os.path.join(step_dir, 'left_event'), exist_ok=False)
        # os.makedirs(os.path.join(step_dir, 'right_event'), exist_ok=False)

        frame0 = inputs['frame0'].to(device)
        frame1 = inputs['frame1'].to(device)
        print(frame0.shape)
        b,c,h,w = frame0.shape
        
        frame0 = frame0[:,:,0:h//4*4,0:w//4*4]
        frame1 = frame1[:,:,0:h//4*4,0:w//4*4]
        for idx in range(len(inputs['event0'])):
            left_event = inputs['event0'][idx][:,:,0:h//4*4,0:w//4*4].to(device)
            right_event = inputs['event1'][idx][:,:,0:h//4*4,0:w//4*4].to(device)
            left_event_cnt = inputs['cnt0'][idx][:,:,0:h//4*4,0:w//4*4].to(device) # only for visualization
            right_event_cnt = inputs['cnt1'][idx][:,:,0:h//4*4,0:w//4*4].to(device) # only for visualization
            gt = inputs['gt'][idx][:,:,0:h//4*4,0:w//4*4].to(device)

            # print(frame0.shape)
            # print(frame1.shape)
            # print(left_event.shape)
            # print(right_event.shape)

            with torch.no_grad():
                start = time.time()
                pred_frame = model(frame0, frame1, left_event, right_event)
                end = time.time()
            pred_frame = torch.clamp(pred_frame,0.0,1.0)



            psnr = compute_psnr(pred_frame, gt)
            lpips = compute_lpips(pred_frame, gt)
            ssim = compute_ssim(tensor2np(pred_frame.detach()[0]), tensor2np(gt.detach()[0]))
            timee = (end-start)
            metric_track.update('psnr', psnr.item())
            metric_track.update('lpips', lpips.item())
            metric_track.update('ssim', ssim.item())
            metric_track.update('time', timee)


            fframe0 = tensor2np(frame0.detach()[0])
            fframe1 = tensor2np(frame1.detach()[0])
            gt = tensor2np(gt.detach()[0])
            pred_frame = tensor2np(pred_frame.detach()[0])
            
            cv2.imwrite(os.path.join(step_dir, 'gt', f'gt{idx:09d}.png'), gt[:, :, [2, 1, 0]])
            cv2.imwrite(os.path.join(step_dir, 'pred_frame', f'pred_frame{idx:09d}.png'), pred_frame[:, :, [2, 1, 0]])
            
            a = vis.plot_event_cnt(left_event_cnt[0].cpu().numpy().transpose(1, 2, 0), is_save=True, path=os.path.join(step_dir, 'left_event', f'left_event{idx:09d}.png'), use_opencv=False)
            b = vis.plot_event_cnt(right_event_cnt[0].cpu().numpy().transpose(1, 2, 0), is_save=True, path=os.path.join(step_dir, 'right_event', f'right_event{idx:09d}.png'), use_opencv=False)
            cv2.imwrite(os.path.join(step_dir, 'left_event', f'left_event{idx:09d}.png'), a[:, :, [2, 1, 0]])
            cv2.imwrite(os.path.join(step_dir, 'right_event', f'right_event{idx:09d}.png'), b[:, :, [2, 1, 0]])

            # vis.plot_frame(torch2frame(frame0[0]), is_save=True, path=os.path.join(step_dir, 'frame0.png'))
            # vis.plot_frame(torch2frame(frame1[0]), is_save=True, path=os.path.join(step_dir, 'frame1.png'))
            # vis.plot_frame(torch2frame(gt[0]), is_save=True, path=os.path.join(step_dir, 'gt.png'))
            # vis.plot_frame(tensor2np(pred_frame.detach()[0]), is_save=True, path=os.path.join(step_dir, 'pred_frame.png'))
            

        cv2.imwrite(os.path.join(step_dir, 'left_frame.png'), fframe0[:, :, [2, 1, 0]])
        cv2.imwrite(os.path.join(step_dir, 'right_frame.png'), fframe1[:, :, [2, 1, 0]])




    result = metric_track.result()
    all_data = metric_track.all_data()
    logger.log_dict(result, 'evaluation results')
    logger.log_dict(all_data, 'all data')



import sys

import torch
import torch.nn as nn
import numpy as np


def get_model_complexity_info(model, input_res,
                              print_per_layer_stat=True,
                              as_strings=True,
                              input_constructor=None, ost=sys.stdout):
    assert type(input_res) is tuple
    assert len(input_res) >= 2
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count()
    if input_constructor:
        input = input_constructor(input_res)
        _ = flops_model(**input)
    else:
        try:
            batch = torch.ones(()).new_empty((1, *input_res),
                                             dtype=next(flops_model.parameters()).dtype,
                                             device=next(flops_model.parameters()).device)
        except StopIteration:
            batch = torch.ones(()).new_empty((1, *input_res))

        _ = flops_model(batch)

    flops_count = abs(flops_model.compute_average_flops_cost())
    params_count = get_model_parameters_number(flops_model)
    if print_per_layer_stat:
        print_model_with_flops(flops_model, flops_count, params_count, ost=ost)
    flops_model.stop_flops_count()

    if as_strings:
        return flops_to_string(flops_count), params_to_string(params_count)

    return flops_count, params_count


def flops_to_string(flops, units='GMac', precision=2):
    if units is None:
        if flops // 10**9 > 0:
            return str(round(flops / 10.**9, precision)) + ' GMac'
        elif flops // 10**6 > 0:
            return str(round(flops / 10.**6, precision)) + ' MMac'
        elif flops // 10**3 > 0:
            return str(round(flops / 10.**3, precision)) + ' KMac'
        else:
            return str(flops) + ' Mac'
    else:
        if units == 'GMac':
            return str(round(flops / 10.**9, precision)) + ' ' + units
        elif units == 'MMac':
            return str(round(flops / 10.**6, precision)) + ' ' + units
        elif units == 'KMac':
            return str(round(flops / 10.**3, precision)) + ' ' + units
        else:
            return str(flops) + ' Mac'


def params_to_string(params_num, units=None, precision=2):
    if units is None:
        if params_num // 10 ** 6 > 0:
            return str(round(params_num / 10 ** 6, 2)) + ' M'
        elif params_num // 10 ** 3:
            return str(round(params_num / 10 ** 3, 2)) + ' k'
        else:
            return str(params_num)
    else:
        if units == 'M':
            return str(round(params_num / 10.**6, precision)) + ' ' + units
        elif units == 'K':
            return str(round(params_num / 10.**3, precision)) + ' ' + units
        else:
            return str(params_num)


def print_model_with_flops(model, total_flops, total_params, units='GMac',
                           precision=3, ost=sys.stdout):

    def accumulate_params(self):
        return get_model_parameters_number(self)

    def accumulate_flops(self):
        if is_supported_instance(self):
            return self.__flops__ / model.__batch_counter__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_flops()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops()
        return ', '.join([params_to_string(accumulated_params_num, units='M', precision=precision),
                          '{:.3%} Params'.format(accumulated_params_num / total_params),
                          flops_to_string(accumulated_flops_cost, units=units, precision=precision),
                          '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
                          self.original_extra_repr()])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    print(model, file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    # Adding variables necessary for masked flops computation
    net_main_module.apply(add_flops_mask_variable_or_reset)

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    batches_count = self.__batch_counter__
    flops_sum = 0
    for module in self.modules():
        if is_supported_instance(module):
            flops_sum += module.__flops__

    return flops_sum / batches_count


def start_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)
    self.apply(add_flops_counter_hook_function)


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


def add_flops_mask(module, mask):
    def add_flops_mask_func(module):
        if isinstance(module, torch.nn.Conv2d):
            module.__mask__ = mask
    module.apply(add_flops_mask_func)


def remove_flops_mask(module):
    module.apply(add_flops_mask_variable_or_reset)


# ---- Internal functions
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    output_last_dim = output.shape[-1]  # pytorch checks dimensions, so here we don't care much
    module.__flops__ += int(np.prod(input.shape) * output_last_dim)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    module.affine
    input = input[0]

    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def deconv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    input_height, input_width = input.shape[2:]

    kernel_height, kernel_width = conv_module.kernel_size
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = kernel_height * kernel_width * in_channels * filters_per_channel

    active_elements_count = batch_size * input_height * input_width
    overall_conv_flops = conv_per_position_flops * active_elements_count
    bias_flops = 0
    if conv_module.bias is not None:
        output_height, output_width = output.shape[2:]
        bias_flops = out_channels * batch_size * output_height * output_height
    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_height, output_width = output.shape[2:]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = np.prod(kernel_dims) * in_channels * filters_per_channel

    active_elements_count = batch_size * np.prod(output_dims)

    if conv_module.__mask__ is not None:
        # (b, 1, h, w)
        flops_mask = conv_module.__mask__.expand(batch_size, 1, output_height, output_width)
        active_elements_count = flops_mask.sum()

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0

    if conv_module.bias is not None:

        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        pass
        print('Warning! No positional inputs found for a module, assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):

    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    handle = module.register_forward_hook(batch_counter_hook)
    module.__batch_counter_handle__ = handle


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        module.__batch_counter_handle__.remove()
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        module.__flops__ = 0


MODULES_MAPPING = {
    # convolutions
    torch.nn.Conv1d: conv_flops_counter_hook,
    torch.nn.Conv2d: conv_flops_counter_hook,
    torch.nn.Conv3d: conv_flops_counter_hook,
    # activations
    torch.nn.ReLU: relu_flops_counter_hook,
    torch.nn.PReLU: relu_flops_counter_hook,
    torch.nn.ELU: relu_flops_counter_hook,
    torch.nn.LeakyReLU: relu_flops_counter_hook,
    torch.nn.ReLU6: relu_flops_counter_hook,
    # poolings
    torch.nn.MaxPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool1d: pool_flops_counter_hook,
    torch.nn.AvgPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool2d: pool_flops_counter_hook,
    torch.nn.MaxPool3d: pool_flops_counter_hook,
    torch.nn.AvgPool3d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    torch.nn.BatchNorm1d: bn_flops_counter_hook,
    torch.nn.BatchNorm2d: bn_flops_counter_hook,
    torch.nn.BatchNorm3d: bn_flops_counter_hook,
    # FC
    torch.nn.Linear: linear_flops_counter_hook,
    # Upscale
    torch.nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    torch.nn.ConvTranspose2d: deconv_flops_counter_hook,
}


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING:
        return True
    return False


def add_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            return
        handle = module.register_forward_hook(MODULES_MAPPING[type(module)])
        module.__flops_handle__ = handle


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            module.__flops_handle__.remove()
            del module.__flops_handle__
# --- Masked flops counting


# Also being run in the initialization
def add_flops_mask_variable_or_reset(module):
    if is_supported_instance(module):
        module.__mask__ = None



    # def get_model_total_params(model):
    #     model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #     params = sum([np.prod(p.size()) for p in model_parameters])
    #     return (1.0*params/(1000*1000))

    # model_stage1 = fusion_network.Fusion()
    # model_stage2 = warp_network.WarpingBasedInterpolationModule()
    # model_stage3 = refine_warp_network.RefineWarpModify()
    # model_stage4 = attention_average_network.AttentionAverageModify()

    # print(get_model_total_params(model_stage1))

if __name__ == "__main__":
    # model = EventVFI_Mo6_Dy21.EventVFItest()
    # flop, param = get_model_complexity_info(model,input_res=(3, 448, 256), as_strings=True, print_per_layer_stat=False)
    # print("GFLOPs: {}".format(flop))
    # print("Params: {}".format(param))
    main()

    # GFLOPs: 100.9 GMac
    # Params: 3.69 M