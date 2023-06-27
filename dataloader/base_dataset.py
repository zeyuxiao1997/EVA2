import torch
from torch.utils.data import Dataset
from abc import abstractmethod
import numpy as np
import random


class BaseDataset(Dataset):
    """
    Base class for dataset
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def sample_index(self, num):
        assert num % 2 == 1, 'Not support even number of images'

        count = 0
        while True:
            indexs = [i for i in range(num) if i % 2 == 0]
            sample_index = sorted(random.sample(indexs, 2))
            candidate_index = [i for i in range(sample_index[0], sample_index[1]) if i % 2 == 1]
            pred_index = random.sample(candidate_index, 1)[0]

            if self.check_index(sample_index, pred_index):
                break

            count += 1
            if count == num:
                print(f'Invalid sampled index for {count} times!')
            elif count == 2 * num:
                raise Exception(f'Invalid sampled index for {count} times!\n Stop reading, please check the data!')
        
        return {'sample_index': sample_index, 'pred_index': pred_index}

    def check_index(self, sample_idx, pred_index):
        frame0_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[0])]
        frame1_h5 = self.h5_file['images']['image{:09d}'.format(sample_idx[1])]
        gt_frame_h5 = self.h5_file['images']['image{:09d}'.format(pred_index)]
        events0_idx = frame0_h5.attrs['event_idx']
        events1_idx = frame1_h5.attrs['event_idx']
        events_gt_idx = gt_frame_h5.attrs['event_idx']

        return (events0_idx < events_gt_idx) and (events1_idx > events_gt_idx)

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

        return torch.from_numpy(frame[:, :, (2, 1, 0)].astype(np.uint8)).permute(2, 0, 1).float()/255