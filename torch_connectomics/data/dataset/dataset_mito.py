from __future__ import print_function, division
import os, sys
import numpy as np
import random
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data

# use image augmentation
from .augmentation import IntensityAugment, simpleaug_train_produce
from .augmentation import apply_elastic_transform, apply_deform

# sample from bounding box
from .T_bbox_mito import process_bbox

# -- 0. utils --
def countVolume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def cropVolume(data, sz, st=[0,0,0]): # C*D*W*H, C=1
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], \
            st[2]:st[2]+sz[2]]      

# -- 1.0 dataset -- 
# dataset class for synaptic cleft inputs
class MitoDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self,
                 volume, 
                 label = None,
                 vol_input_size = (8,64,64),
                 vol_label_size = None,
                 sample_stride = (1,1,1),
                 data_aug = False,
                 border = None,
                 mode = 'train'):
        
        self.mode = mode

        # data format
        self.input = volume
        self.label = label
        self.border = border
        self.data_aug = data_aug # data augmentation
        
        # samples, channels, depths, rows, cols
        self.input_size = [np.array(x.shape) for x in self.input] # volume size, could be multi-volume input
        self.vol_input_size = np.array(vol_input_size) # model input size
        self.vol_label_size = np.array(vol_label_size) # model label size

        # compute number of samples for each dataset (multi-volume input)
        self.sample_stride = np.array(sample_stride, dtype=np.float32)
        self.sample_size = [ countVolume(self.input_size[x], self.vol_input_size, np.array(self.sample_stride)) \
                            for x in range(len(self.input_size))]
        #total number of possible inputs for each volume
        self.sample_num = np.array([np.prod(x) for x in self.sample_size])
        self.sample_num_a = np.sum(self.sample_num)
        self.sample_num_c = np.cumsum([0] + list(self.sample_num))
        #print(self.sample_num_c)
        assert self.sample_num_c[-1] == self.sample_num_a

        '''
        Image augmentation
        1. self.simple_aug: Simple augmentation, including mirroring and transpose
        2. self.intensity_aug: Intensity augmentation
        '''
        if self.data_aug:
            self.simple_aug = simpleaug_train_produce(model_io_size = self.vol_input_size)
            self.intensity_aug = IntensityAugment(mode='mix', skip_ratio=0.5, CONTRAST_FACTOR=0.1, BRIGHTNESS_FACTOR=0.1)

        # for test
        self.sample_size_vol = [np.array([np.prod(x[1:3]),x[2]]) for x in self.sample_size]

    def __getitem__(self, index):

        if self.mode == 'train':
            # 1. get volume size
            vol_size = self.vol_input_size
            # if self.data_aug is not None: # augmentation
            #     self.data_aug.getParam() # get augmentation parameter
            #     vol_size = self.data_aug.aug_warp[0]
            # train: random sample based on vol_size
            # test: sample based on index

            # reject no-synapse samples with a probability of p 
            seed = np.random.RandomState(index)
            while True:
                pos = self.getPosSeed(vol_size, seed)
                out_label = cropVolume(self.label[pos[0]], vol_size, pos[1:])
                if np.sum(out_label) > 100:
                    break
                else:
                    #if random.random() > 0.75:
                    if random.random() > 0.75:    
                        break       

            # pos = self.getPos(vol_size, index)

            # 2. get input volume
            out_input = cropVolume(self.input[pos[0]], vol_size, pos[1:])
            out_border = cropVolume(self.border[pos[0]], vol_size, pos[1:])
            # out_label = cropVolume(self.label[pos[0]], vol_size, pos[1:])

            # 3. augmentation
            if self.data_aug: # augmentation
                #if random.random() > 0.5:
                #    out_input, out_label = apply_elastic_transform(out_input, out_label)    
                out_input, out_label, out_border = self.simple_aug.multi_mask([out_input, out_label, out_border])
                #out_input = self.intensity_aug.augment(out_input)
                if random.random() > 0.5: out_input = apply_deform(out_input)
            
            # 4. class weight
            # add weight to classes to handle data imbalance
            # match input tensor shape
            out_input = torch.from_numpy(out_input.copy())
            out_label = torch.from_numpy(out_label.copy())
            out_border = torch.from_numpy(out_border.copy())

            weight_factor = out_label.float().sum() / torch.prod(torch.tensor(out_label.size()).float())
            weight_factor = torch.clamp(weight_factor, min=1e-4, max=0.5)
            # the fraction of synaptic cleft pixels, can be 0
            alpha = 4.0
            beta = 5.0
            weight = alpha * out_label*(1-weight_factor)/weight_factor + (1-out_label)
            weight = weight + beta*out_border

            # include the channel dimension
            out_input = out_input.unsqueeze(0)
            out_label = out_label.unsqueeze(0)
            weight = weight.unsqueeze(0)

            return out_input, out_label, weight, weight_factor

        elif self.mode == 'test':
            # 1. get volume size
            vol_size = self.vol_input_size  
            # test mode
            pos = self.getPosTest(index)
            out_input = cropVolume(self.input[pos[0]], vol_size, pos[1:])
            out_input = torch.from_numpy(out_input.copy())
            out_input = out_input.float()
            out_input = out_input.unsqueeze(0)

            return pos, out_input  

    def __len__(self): # number of possible position
        return self.sample_num_a
    
    def getPosDataset(self, index):
        return np.argmax(index<self.sample_num_c)-1 # which dataset

    def getPos(self, vol_size, index):
        pos = [0,0,0,0]
        # support random sampling using the same 'index'
        seed = np.random.RandomState(index)
        did = self.getPosDataset(seed.randint(self.sample_num_a))
        pos[0] = did
        tmp_size = countVolume(self.input_size[did], vol_size, np.array(self.sample_stride))
        pos[1:] = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        return pos

    def index2zyx(self, index): # for test
        # int division = int(floor(.))
        pos = [0,0,0,0]
        did = self.getPosDataset(index)
        pos[0] = did
        index2 = index - self.sample_num_c[did]
        pos[1:] = self.getPosLocation(index2, self.sample_size_vol[did])
        return pos 

    def getPosLocation(self, index, sz):
        # sz: [y*x, x]
        pos = [0,0,0]
        pos[0] = np.floor(index/sz[0])
        pz_r = index % sz[0]
        pos[1] = np.floor(pz_r/sz[1])
        pos[2] = pz_r % sz[1]
        return pos 

    def getPosTest(self, index):
        pos = self.index2zyx(index)
        for i in range(1,4):
            if pos[i] != self.sample_size[pos[0]][i-1]-1:
                pos[i] = int(pos[i] * self.sample_stride[i-1])
            else:
                pos[i] = int(self.input_size[pos[0]][i-1]-self.vol_input_size[i-1])
        return pos

    def getPosSeed(self, vol_size, seed):
        pos = [0,0,0,0]
        did = self.getPosDataset(seed.randint(self.sample_num_a))
        pos[0] = did
        tmp_size = countVolume(self.input_size[did], vol_size, np.array(self.sample_stride))
        pos[1:] = [np.random.randint(tmp_size[x]) for x in range(len(tmp_size))]
        return pos    


class BboxMitoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 bbox_text,
                 g_text,
                 b_text,
                 data_aug = False,
                 use_gt = True,
                 mode = 'train'):

        self.use_gt = use_gt
        self.mode = mode
        print('Mode: ' + self.mode)
        self.data_aug = data_aug

        self.bbox = []

        bbox_file = open(bbox_text, 'r')
        bbox_list = [y.strip().split(' ') for y in bbox_file.readlines()]
        #print(bbox_list[0])
        #print(list(map(int, bbox_list[0])))
        #exit(0)
        for i in range(len(bbox_list)): 
            #print(i)
            #print(bbox_list[i])
            self.bbox.append(list(map(int, bbox_list[i]))[:6])
        self.bbox = np.array(self.bbox).astype(int)

        if self.mode == 'train':
            self.g_list = list(np.loadtxt(g_text, dtype=int))
            self.b_list = list(np.loadtxt(b_text, dtype=int))
            self.total = self.g_list + self.b_list
            print('num of TP samples: ',len(self.g_list))
            print('num of FP samples: ',len(self.b_list))
            print('number of samples with label: ', len(self.total))
        print('total number of samples: ', len(self.bbox))

        if self.data_aug:
            #self.simple_aug = simpleaug_train_produce(model_io_size = self.vol_input_size)
            self.intensity_aug = IntensityAugment(mode='mix', skip_ratio=0.5, CONTRAST_FACTOR=0.1, BRIGHTNESS_FACTOR=0.1)

    def gaussian_weight(self, sz):
        zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                         np.linspace(-1,1,sz[1], dtype=np.float32),
                         np.linspace(-1,1,sz[2], dtype=np.float32), indexing='ij')

        dd = np.sqrt(zz*zz + yy*yy + xx*xx)
        sigma, mu = 0.7, 0.0
        ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))  
        return torch.from_numpy(ww.copy())         

    def __getitem__(self, index):

        # 1. 
        # 2. get input volume
        input_bbox = list(self.bbox[index])
        gt = (index in self.g_list)
        #gt = True
        if self.use_gt:
            out_input, out_label = process_bbox(input_bbox, gt, use_gt=True) # out_lable: (3,z,y,x)
        else:
            out_input, out_label = process_bbox(input_bbox, gt, use_gt=False)
        
        if self.mode == 'train':
            # 3. augmentation
            if self.data_aug: # augmentation
                #if random.random() > 0.5:
                #    out_input, out_label = apply_elastic_transform(out_input, out_label)    
                #out_input, out_label = self.simple_aug(out_input, out_label)
                #out_input = self.intensity_aug.augment(out_input)
                if random.random() > 0.5: out_input = apply_deform(out_input)
            
            # 4. class weight
            # add weight to classes to handle data imbalance
            # match input tensor shape
            out_input = torch.from_numpy(out_input.copy())
            out_label = torch.from_numpy(out_label.copy())
            weight_factor = out_label.float().sum() / torch.prod(torch.tensor(out_label.size()).float())
            weight_factor = torch.clamp(weight_factor, min=1e-4)
            # the fraction of synaptic cleft pixels, can be 0
            alpha = 3.0
            weight = alpha * out_label*(1-weight_factor)/weight_factor + (1.0-out_label)

            # include the channel dimension
            out_input = out_input.unsqueeze(0)
            out_label = out_label.unsqueeze(0)
            weight = weight.unsqueeze(0)

            #print(index, input_bbox, (index in self.g_list))
            return out_input, out_label, weight, weight_factor, int(gt), input_bbox

        # elif self.mode == 'validation':
        #     out_input = torch.from_numpy(out_input.copy())
        #     out_label = torch.from_numpy(out_label.copy())
        #     out_input = out_input.unsqueeze(0)
        #     return out_input, out_label

        # elif self.mode == 'test':    
        #     out_input = torch.from_numpy(out_input.copy())
        #     out_input = out_input.unsqueeze(0)
        #     return out_input

    def __len__(self): # number of possible position
        return self.bbox.shape[0]    


def collate_fn_bbox(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    out_input, out_label, weights, weight_factor, gt, bbox = zip(*batch)
    out_input = torch.stack(out_input, 0)
    out_label = torch.stack(out_label, 0)
    weights = torch.stack(weights, 0)

    weight_factor = np.stack(weight_factor, 0)

    return out_input, out_label, weights, weight_factor, gt, bbox                
