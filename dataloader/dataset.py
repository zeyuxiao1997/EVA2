import torch
import os
from glob import glob
import h5py
import cv2
import numpy as np
import random
import yaml
# local modules
from dataloader.base_dataset_old import BaseDataset
from dataloader.encodings import *

# from base_dataset import BaseDataset
# from encodings import *


class H5Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

        self.data_path = config['data']['path']
        self.h5_file_path = sorted(glob(os.path.join(self.data_path, '*.h5')))

    def __getitem__(self, index):
        # index = index+1320+230
        # index = index+50330+540+2300+3510+2740+1110+390+1360
        seed = random.randint(0, 2**32)

        self.h5_file = h5py.File(self.h5_file_path[index], 'r')
        self.sensor_resolution = self.h5_file.attrs['sensor_resolution']
        self.time_bins = self.config['data']['time_bins']
        num_imgs = self.h5_file.attrs['num_imgs']
        idx = self.sample_index(num_imgs)
        sample_idx, pred_idx = idx['sample_index'], idx['pred_index']

        # RGB and aumentation
        frame0_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[0])]
        frame1_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[1])]
        gt_frame_h5 = self.h5_file['images']['image{:09d}'.format(pred_idx)]
        frame0 = frame0_h5[:]
        frame1 = frame1_h5[:]
        gt_frame = gt_frame_h5[:]
        if self.config['data_augment']['enabled']:
            frame0 = self.augment_frame(frame0, seed)
            frame1 = self.augment_frame(frame1, seed)
            gt_frame = self.augment_frame(gt_frame, seed)
        frame0_torch = self.frame_formatting(frame0)
        frame1_torch = self.frame_formatting(frame1)
        gt_frame_torch = self.frame_formatting(gt_frame)

        # events
        events0_idx = [frame0_h5.attrs['event_idx'], gt_frame_h5.attrs['event_idx']]
        events1_idx = [gt_frame_h5.attrs['event_idx'], frame1_h5.attrs['event_idx']]
        events0 = self.get_events(events0_idx[0], events0_idx[1])
        events1 = self.get_events(events1_idx[0], events1_idx[1])
        if self.config['data_augment']['enabled']:
            events0 = self.augment_event(events0, seed)
            events1 = self.augment_event(events1, seed)
        events0_torch = self.event_formatting(events0)
        events1_torch = self.event_formatting(events1)

        # convert to event voxel
        voxel0 = self.create_voxel_encoding(events0_torch)
        voxel1 = self.create_voxel_encoding(events1_torch)

        # add hot pixel and noise
        if self.config['add_hot_pixels']['enabled']:
            self.add_hot_pixels_to_voxel(voxel0, 
                                         hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                         hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
            self.add_hot_pixels_to_voxel(voxel1, 
                                         hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                         hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
        if self.config['add_noise']['enabled']:
            voxel0 = self.add_noise_to_voxel(voxel0,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction'])
            voxel1 = self.add_noise_to_voxel(voxel1,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction'])

        item = {'input':{'frame0': frame0_torch,
                         'frame1': frame1_torch,
                         'event0': voxel0,
                         'event1': voxel1
                         },
                'gt': gt_frame_torch
                }

        return item
        
    def __len__(self):

        return len(self.h5_file_path)

    def get_events(self, idx0, idx1):
        # event的数据是4维， idx0到idx1之间的值
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1]

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0)

    def create_voxel_encoding(self, events):
        
        return events_to_voxel_torch(events[0], events[1], events[2], events[3], B=self.time_bins, sensor_size=self.sensor_resolution)

    def augment_event(self, events, seed):
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]
        seed_H, seed_W, seed_P = seed, seed + 1, seed + 2

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    xs = self.sensor_resolution[1] - 1 - xs
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ys = self.sensor_resolution[0] - 1 - ys
            elif mechanism == 'Polarity':
                random.seed(seed_P)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ps = ps * -1

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0)

    def augment_frame(self, img, seed):
        seed_H, seed_W = seed, seed + 1

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 1)
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 0)

        return img

    @staticmethod
    def add_hot_pixels_to_voxel(voxel, hot_pixel_std=1.0, hot_pixel_fraction=0.001):
        num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
        x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
        y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
        for i in range(num_hot_pixels):
            voxel[..., :, y[i], x[i]] += random.gauss(0, hot_pixel_std)
    
    @staticmethod
    def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
        noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand_like(voxel) >= noise_fraction
            noise.masked_fill_(mask, 0)
        return voxel + noise 



class H5Dataset_Val(BaseDataset):
    def __init__(self, config):
        super().__init__(config)

        self.data_path = config['data']['path']
        self.h5_file_path = sorted(glob(os.path.join(self.data_path, '*.h5')))

    def __getitem__(self, index):
        # index = index+1320+230
        # index = index+50330+540+2300+3510+2740+1110+390+1360
        seed = random.randint(0, 2**32)

        self.h5_file = h5py.File(self.h5_file_path[index], 'r')
        self.sensor_resolution = self.h5_file.attrs['sensor_resolution']
        self.time_bins = self.config['data']['time_bins']
        num_imgs = self.h5_file.attrs['num_imgs']
        idx = self.sample_index(num_imgs)
        sample_idx, pred_idx = idx['sample_index'], idx['pred_index']

        # RGB and aumentation
        frame0_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[0])]
        frame1_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[1])]
        gt_frame_h5 = self.h5_file['images']['image{:09d}'.format(pred_idx)]
        frame0 = frame0_h5[:]
        frame1 = frame1_h5[:]
        gt_frame = gt_frame_h5[:]
        if self.config['data_augment']['enabled']:
            frame0 = self.augment_frame(frame0, seed)
            frame1 = self.augment_frame(frame1, seed)
            gt_frame = self.augment_frame(gt_frame, seed)
        frame0_torch = self.frame_formatting(frame0)
        frame1_torch = self.frame_formatting(frame1)
        gt_frame_torch = self.frame_formatting(gt_frame)

        # events
        events0_idx = [frame0_h5.attrs['event_idx'], gt_frame_h5.attrs['event_idx']]
        events1_idx = [gt_frame_h5.attrs['event_idx'], frame1_h5.attrs['event_idx']]
        events0 = self.get_events(events0_idx[0], events0_idx[1])
        events1 = self.get_events(events1_idx[0], events1_idx[1])
        if self.config['data_augment']['enabled']:
            events0 = self.augment_event(events0, seed)
            events1 = self.augment_event(events1, seed)
        events0_torch = self.event_formatting(events0)
        events1_torch = self.event_formatting(events1)

        # convert to event voxel
        voxel0 = self.create_voxel_encoding(events0_torch)
        voxel1 = self.create_voxel_encoding(events1_torch)

        # add hot pixel and noise
        if self.config['add_hot_pixels']['enabled']:
            self.add_hot_pixels_to_voxel(voxel0, 
                                         hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                         hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
            self.add_hot_pixels_to_voxel(voxel1, 
                                         hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                         hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
        if self.config['add_noise']['enabled']:
            voxel0 = self.add_noise_to_voxel(voxel0,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction'])
            voxel1 = self.add_noise_to_voxel(voxel1,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction'])

        # item = {'input':{'frame0': frame0_torch,
        #                  'frame1': frame1_torch,
        #                  'event0': voxel0,
        #                  'event1': voxel1
        #                  },
        #         'gt': gt_frame_torch
        #         }

        return frame0_torch,frame1_torch,voxel0,voxel1,gt_frame_torch
        
    def __len__(self):

        return len(self.h5_file_path)

    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1]

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0)

    def create_voxel_encoding(self, events):
        
        return events_to_voxel_torch(events[0], events[1], events[2], events[3], B=self.time_bins, sensor_size=self.sensor_resolution)

    def augment_event(self, events, seed):
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]
        seed_H, seed_W, seed_P = seed, seed + 1, seed + 2

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    xs = self.sensor_resolution[1] - 1 - xs
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ys = self.sensor_resolution[0] - 1 - ys
            elif mechanism == 'Polarity':
                random.seed(seed_P)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    ps = ps * -1

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0)

    def augment_frame(self, img, seed):
        seed_H, seed_W = seed, seed + 1

        for i, mechanism in enumerate(self.config['data_augment']['augment']):
            if mechanism == 'Horizontal':
                random.seed(seed_H)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 1)
            elif mechanism == 'Vertical':
                random.seed(seed_W)
                if random.random() < self.config['data_augment']['augment_prob'][i]:
                    img = np.flip(img, 0)

        return img

    @staticmethod
    def add_hot_pixels_to_voxel(voxel, hot_pixel_std=1.0, hot_pixel_fraction=0.001):
        num_hot_pixels = int(hot_pixel_fraction * voxel.shape[-1] * voxel.shape[-2])
        x = torch.randint(0, voxel.shape[-1], (num_hot_pixels,))
        y = torch.randint(0, voxel.shape[-2], (num_hot_pixels,))
        for i in range(num_hot_pixels):
            voxel[..., :, y[i], x[i]] += random.gauss(0, hot_pixel_std)
    
    @staticmethod
    def add_noise_to_voxel(voxel, noise_std=1.0, noise_fraction=0.1):
        noise = noise_std * torch.randn_like(voxel)  # mean = 0, std = noise_std
        if noise_fraction < 1.0:
            mask = torch.rand_like(voxel) >= noise_fraction
            noise.masked_fill_(mask, 0)
        return voxel + noise 



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    def rescale(input):
        max = input.max()
        min = input.min()
        
        return (input-min) / (max - min + 1e-6)

    file = 'dataloader/test.yml'
    file = 'test.yml'

    with open(file) as fid:
        config = yaml.load(fid, Loader=yaml.FullLoader)
    
    dataset = H5Dataset(config)

    train_dataloader = DataLoader(dataset,
                        batch_size=8,
                        shuffle=False,
                        num_workers=0,
                        drop_last=True)
    iteration = 0
    for _, batch in enumerate(train_dataloader, 0):
        iteration += 1 

        frame0 = batch['input']['frame0']
        frame1 = batch['input']['frame1']
        event0 = batch['input']['event0']
        event1 = batch['input']['event1']
        target = batch['gt']
        print(event0, event1)
        # print(iteration, frame0.shape)
        # frame0, frame1, event0, event1, target = frame0.float().cuda(), frame1.float().cuda(), event0.float().cuda(), event1.float().cuda(), target.float().cuda()