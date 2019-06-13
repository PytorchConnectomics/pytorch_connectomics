from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

from .misc import *

# -- 1.0 dataset -- 
# dataset class for synaptic cleft inputs
class BaseDataset(torch.utils.data.Dataset):
    """
    # sample_input_size: sample input size
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 mode='train'):

        self.mode = mode
        # for partially labeled data
        # m1 (no): sample chunks with over certain percentage
        #   = online version: rejection sampling can be slow
        #   = offline version: need to keep track of the mask
        # self.label_ratio = label_ratio
        # m2: make sure the center is labeled

        # data format
        self.input = volume
        self.label = label
        self.augmentor = augmentor  # data augmentation

        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.input]  # volume size, could be multi-volume input
        self.sample_input_size = np.array(sample_input_size)  # model input size
        self.sample_label_size = np.array(sample_label_size)  # model label size

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride, dtype=np.float32)
        self.sample_size = [count_volume(self.input_size[x], self.sample_input_size, np.array(self.sample_stride))
                            for x in range(len(self.input_size))]
        # total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        # check partial label
        self.label_invalid = [False]*len(self.sample_num)
        if self.label is not None:
            for i in range(len(self.sample_num)):
                seg_id = np.array([-1]).astype(self.label[i].dtype)[0]
                seg = self.label[i][sample_label_size[0]//2:-sample_label_size[0]//2,\
                                    sample_label_size[1]//2:-sample_label_size[1]//2,\
                                    sample_label_size[2]//2:-sample_label_size[2]//2]

                if np.any(seg == seg_id):
                    print('dataset %d: needs mask for invalid region'%(i))
                    self.label_invalid[i] = True
                    self.sample_num[i] = np.count_nonzero(seg != seg_id)

        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))

        '''
        Image augmentation
        1. self.simple_aug: Simple augmentation, including mirroring and transpose
        2. self.intensity_aug: Intensity augmentation
        '''
        if mode=='test': # for test
            self.sample_size_vol = [np.array([np.prod(x[1:3]), x[2]]) for x in self.sample_size]

    def __getitem__(self, index):
        raise NotImplementedError("Need to implement getitem() !")

    def __len__(self):  # number of possible position
        return self.sample_num_a

    def get_pos_dataset(self, index):
        return np.argmax(index < self.sample_num_c) - 1  # which dataset

    def get_pos(self, vol_size, index):
        pos = [0, 0, 0, 0]
        # support random sampling using the same 'index'
        seed = np.random.RandomState(index)
        did = self.get_pos_dataset(seed.randint(self.sample_num_a))
        pos[0] = did
        tmp_size = count_volume(self.input_size[did], vol_size, np.array(self.sample_stride))
        pos[1:] = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        return pos

    def index2zyx(self, index):  # for test
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = self.get_pos_dataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self.get_pos_location(index2, self.sample_size_vol[did])
        return pos

    def get_pos_location(self, index, sz):
        # index -> z,y,x
        # sz: [y*x, x]
        pos = [0, 0, 0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = np.floor(pz_r/sz[1])
        pos[2] = pz_r % sz[1]
        return pos

    def get_pos_test(self, index):
        pos = self.index2zyx(index)
        for i in range(1, 4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = int(pos[i] * self.sample_stride[i-1])
            else:
                pos[i] = int(self.input_size[pos[0]][i-1]-self.sample_input_size[i-1])
        return pos

    def get_pos_seed(self, vol_size, seed):
        pos = [0, 0, 0, 0]
        # pick a dataset
        did = self.get_pos_dataset(seed.randint(self.sample_num_a))
        pos[0] = did
        # pick a position
        tmp_size = count_volume(self.input_size[did], vol_size, np.array(self.sample_stride))
        tmp_pos = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        if self.label_invalid[did]:
            # need to make sure the center is valid
            seg_bad = np.array([-1]).astype(self.label[did].dtype)[0]
            #print(seg_bad,self.label[did][tmp_pos[0]+vol_size[0]//2,tmp_pos[1]+vol_size[1]//2,tmp_pos[2]+vol_size[2]//2])
            while self.label[did][tmp_pos[0]+vol_size[0]//2,\
                                  tmp_pos[1]+vol_size[1]//2,\
                                  tmp_pos[2]+vol_size[2]//2] == seg_bad:
                tmp_pos = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        pos[1:] = tmp_pos 
        return pos