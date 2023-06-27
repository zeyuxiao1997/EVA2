import torch
import os
from glob import glob
import h5py
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
import yaml
# local modules
from dataloader_TimeLens_weng_new.encodings import *


class H5Dataset(Dataset):
    def __init__(self, h5_file_path, config):
        super().__init__()
        self.config = config
        self.h5_file_path = h5_file_path

        self.load_metadata()
        self.compute_indices()

    def load_metadata(self):
        self.h5_file = h5py.File(self.h5_file_path, 'r')
        self.sensor_resolution = self.h5_file.attrs['sensor_resolution']

        self.time_bins = self.config['time_bins']
        self.skip_frame = self.config['skip_frame']
        self.num_imgs = self.h5_file.attrs['num_imgs']

    def compute_indices(self):
        assert self.skip_frame >= 1, 'Skipped frames must >= 1!'
        assert self.num_imgs >= 3, 'Images number must >= 3!'

        self.keyframe_indices = []
        self.predframe_indices = []
        candidates_indices = np.arange(0, self.num_imgs, self.skip_frame+1)
        for left, right in zip(candidates_indices[:-1], candidates_indices[1:]):
            self.keyframe_indices.append([left, right])
            self.predframe_indices.append([pred for pred in range(left+1, right)])

    def __getitem__(self, index):
        seed = random.randint(0, 2**32)

        keyframe_idx, pred_idx = self.keyframe_indices[index], self.predframe_indices[index]

        # RGB and augmentation
        frame0_h5 = self.h5_file['images']['image{:09d}'.format(keyframe_idx[0])]
        frame1_h5 = self.h5_file['images']['image{:09d}'.format(keyframe_idx[1])]
        gt_frame_h5 = [self.h5_file['images']['image{:09d}'.format(idx)] for idx in pred_idx]

        # frame0 = cv2.cvtColor(frame0_h5[:], cv2.COLOR_BGR2GRAY) # convert RGB to gray, only for HQF or E2VID
        # frame1 = cv2.cvtColor(frame1_h5[:], cv2.COLOR_BGR2GRAY)
        # gt_frame = [cv2.cvtColor(img[:], cv2.COLOR_BGR2GRAY) for img in gt_frame_h5]
        frame0 = frame0_h5[:] # convert RGB to gray, for vimeo90k, middlebury, gopro, hsergn
        frame1 = frame1_h5[:]
        gt_frame = [img[:] for img in gt_frame_h5]

        if self.config['data_augment']['enabled']:
            frame0 = self.augment_frame(frame0, seed)
            frame1 = self.augment_frame(frame1, seed)
            gt_frame = [self.augment_frame(img, seed) for img in gt_frame]
        frame0_torch = self.frame_formatting(frame0)
        frame1_torch = self.frame_formatting(frame1)
        gt_frame_torch = [self.frame_formatting(img) for img in gt_frame]

        # events
        events0_idx = [[frame0_h5.attrs['event_idx'], img.attrs['event_idx']] for img in gt_frame_h5]
        events1_idx = [[img.attrs['event_idx'], frame1_h5.attrs['event_idx']] for img in gt_frame_h5]
        events0 = [self.get_events(idx[0], idx[1]) for idx in events0_idx]
        events1 = [self.get_events(idx[0], idx[1]) for idx in events1_idx]

        # tmp = self.get_events(events0_idx[0], events0_idx[1])
        # reverse event for timeLens
        # if tmp.shape[1] != 0:
        events0_reverse = [self.reverse_events(events) for events in events0]
        # else:
        #     events0_reverse = events0

        if self.config['data_augment']['enabled']:
            events0 = [self.augment_event(event, seed) for event in events0]
            events1 = [self.augment_event(event, seed) for event in events1]
            events0_reverse = [self.augment_event(event, seed) for event in events0_reverse]

        events0_torch = [self.event_formatting(event) for event in events0]
        events1_torch = [self.event_formatting(event) for event in events1]
        events0_reverse = [self.event_formatting(event) for event in events0_reverse]

        # convert to event voxel
        voxel0 = [self.create_voxel_encoding(event_torch) for event_torch in events0_torch]
        voxel1 = [self.create_voxel_encoding(event_torch) for event_torch in events1_torch]
        voxel0_reverse = [self.create_voxel_encoding(event_torch) for event_torch in events0_reverse]

        # cnt0 = [self.create_cnt_encoding(event_torch) for event_torch in events0_torch]
        # cnt1 = [self.create_cnt_encoding(event_torch) for event_torch in events1_torch]

        # add hot pixel and noise
        if self.config['add_hot_pixels']['enabled']:
            for v0, v1 in zip(voxel0, voxel1):
                self.add_hot_pixels_to_voxel(v0, 
                                             hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                             hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
                self.add_hot_pixels_to_voxel(v1, 
                                             hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                             hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])
                self.add_hot_pixels_to_voxel(voxel0_reverse, 
                                         hot_pixel_std=self.config['add_hot_pixels']['hot_pixel_std'],
                                         hot_pixel_fraction=self.config['add_hot_pixels']['hot_pixel_fraction'])

        if self.config['add_noise']['enabled']:
            voxel0 = [self.add_noise_to_voxel(voxel,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction']) for voxel in voxel0]
            voxel1 = [self.add_noise_to_voxel(voxel,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction']) for voxel in voxel1]
            voxel0_reverse = self.add_noise_to_voxel(voxel0_reverse,
                                             noise_std=self.config['add_noise']['noise_std'],
                                             noise_fraction=self.config['add_noise']['noise_fraction'])

        item = {'frame0': frame0_torch,
                'frame1': frame1_torch,
                'event0': voxel0,
                'event1': voxel1,
                # 'cnt0': cnt0,
                # 'cnt1': cnt1,
                'event0_reverse': voxel0_reverse,
                'gt': gt_frame_torch,
                'name': self.h5_file_path
                }

        return item


        
    def __len__(self):

        return len(self.predframe_indices)


    # reverse Event
    @staticmethod
    def reverse_events(events):
        """
        events: np.array, 4xN, [xs, ys, ts, ps]
        return: reversed events, np.array, 4xN, [xs, ys, ts, ps]

        Polarities of the events reversed.

                          (-)       (+) 
        --------|----------|---------|------------|----> time
           t_start        t_1       t_2        t_end

                          (+)       (-) 
        --------|----------|---------|------------|----> time
                0    (t_end-t_2) (t_end-t_1) (t_end-t_start)  
        """
        events = events.transpose(1, 0) # 4 x N --> N x 4
        start_time, end_time = events[0, 2], events[-1, 2]
        reversed_events = events[::-1, :]
        reversed_events[:, 2] = start_time + (end_time - reversed_events[:, 2])
        reversed_events[:, 3] = reversed_events[:, 3] * -1

        return reversed_events.transpose(1, 0)




    def get_events(self, idx0, idx1):
        xs = self.h5_file['events/xs'][idx0:idx1]
        ys = self.h5_file['events/ys'][idx0:idx1]
        ts = self.h5_file['events/ts'][idx0:idx1]
        ps = self.h5_file['events/ps'][idx0:idx1]

        # ps = ps.astype(np.int); ps[ps==0] = -1 # uncomment this line for HQF

        return np.concatenate((xs[np.newaxis, ...], ys[np.newaxis, ...], ts[np.newaxis, ...], ps[np.newaxis, ...]), axis=0)

    def create_voxel_encoding(self, events):
        
        return events_to_voxel_torch(events[0], events[1], events[2], events[3], B=self.time_bins, sensor_size=self.sensor_resolution)

    def create_cnt_encoding(self, events):
        """
        events: torch.tensor, 4xN [x, y, t, p]
        
        return: count: torch.tensor, 2 x H x W
        """
        xs, ys, ts, ps = events[0], events[1], events[2], events[3]

        return events_to_channels(xs, ys, ps, sensor_size=self.sensor_resolution)

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

    @staticmethod
    def event_formatting(events):
        
        xs = torch.from_numpy(events[0].astype(np.float32))
        ys = torch.from_numpy(events[1].astype(np.float32))
        ts = torch.from_numpy(events[2].astype(np.float32))
        ps = torch.from_numpy(events[3].astype(np.float32))
        # ts = (ts - ts[0]) / (ts[-1] - ts[0])
        return torch.stack([xs, ys, ts, ps])

    @staticmethod
    def frame_formatting(frame):
        if len(frame.shape) == 2:
            return torch.from_numpy(frame).float() / 255
        elif len(frame.shape) == 3:
            return torch.from_numpy(frame[:, :, (2, 1, 0)]).permute(2, 0, 1).float() / 255
        else:
            raise Exception('Wrong frame!')
