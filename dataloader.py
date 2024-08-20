# this file contains functions required for loading the data for training or evaluation

import numpy as np
from torch.utils.data import Dataset

from flags import *
from os import listdir
from os import path as osp
import torch
import h5py

def load_dataset():
    # loading data
    if (DATASET == 'moving_mnist'):
        dataset = data_moving_mnist(DATA_PATH)
    elif (DATASET == 'dsprites_color'):
        dataset = data_dsprites_color(DATA_PATH)
    elif DATASET == 'dsprites_color_test':
        dataset = data_dsprites_color(DATA_PATH)
    elif DATASET == 'sprites':
        dataset = data_sprites(DATA_PATH)
    elif DATASET == 'sprites_test':
        dataset = data_sprites(DATA_PATH)
    else:
        raise Exception('Invalid Dataset!')

    return dataset

class data_moving_mnist:
    def __init__(self, DATA_PATH):
        DATA_PATH = r'/home/azencot_group/datasets/SMD/moving_mnist_simple'
        self.array = []
        for f in [x for x in listdir(DATA_PATH) if '1d64' in x]:
            self.data = np.load(osp.join(DATA_PATH, f))
            self.arr = np.reshape(self.data['arr_0'], [10000, 20, 64, 64])
            for i in range(10000):
                self.array.append(np.reshape(self.arr[i, :2*NUM_FRAMES:2, ], [NUM_FRAMES, NUM_INPUT_CHANNELS, H, W]))

    def __len__(self):
        return self.array.__len__()

    def __getitem__(self, index):
        return self.array[index]


class data_dsprites_color(Dataset): # Shape:  N, 8, 3, 32, 32
    def __init__(self, file):
        super(data_dsprites_color, self).__init__()
        self.file = h5py.File(file, 'r')
        self.n_videos = np.asarray(self.file.get('data')).astype(float) / 255.0

    def __getitem__(self, index):
        input = self.n_videos[index]
        return input.astype('float32')

    def __len__(self):
        return self.n_videos.shape[0]

class data_sprites(Dataset):
    def __init__(self, data_path, return_labels=False): # N, 8, 64, 64, 3 --reshape--> N, 8, 3, 64, 64
        super(data_sprites, self).__init__()
        self.data_path = data_path
        self.return_labels = return_labels
        dataset = np.load(self.data_path, allow_pickle=True)
        self.data = dataset['data'].transpose(0, 1, 4, 2, 3).copy()
        self.labels = dataset['labels']
        self.values = dataset['values']
        self.classes = dataset['classes'][()]

    def __getitem__(self, item):
        if self.return_labels:
            return self.data[item], self.labels[item]
        else:
            return self.data[item]

    def __len__(self):
        return len(self.data)