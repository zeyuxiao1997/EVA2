from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
# Local modules
from .h5dataset import H5Dataset


def concatenate_datasets(data_paths, dataset_type, dataset_config):
    dataset_list = []
    print('Concatenating {} datasets'.format(dataset_type))
    for data_path in tqdm(data_paths):
        dataset_list.append(dataset_type(data_path, dataset_config))

    return ConcatDataset(dataset_list)


class H5Dataloader(DataLoader):
    def __init__(self, config):
        self.config = config

        self.data_path = config['path']
        self.h5_file_path = sorted(glob(os.path.join(self.data_path, '*.h5')))
        self.dataset = concatenate_datasets(self.h5_file_path, H5Dataset, self.config['dataset'])

        super().__init__(self.dataset, **self.config['args'])



