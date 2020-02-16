from __future__ import print_function, division
import numpy as np
from scipy.ndimage.morphology import binary_dilation
import random

import torch
import torch.utils.data

from ..utils import seg_to_aff, get_small_seg, get_instance_bd, crop_volume, rebalance_binary_class, relabel
from .dataset import BaseDataset

class AffinityDataset(BaseDataset):
    """PyTorch ddataset class for affinity graph prediction.

    Args:
        volume: input image stacks.
        label: segmentation stacks.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor: data augmentor.
        mode (str): training or inference mode.
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 mode='train', loss_opt=1, loss_weight=0):

        super(AffinityDataset, self).__init__(volume,label,
                                              sample_input_size,
                                              sample_label_size,
                                              sample_stride,
                                              augmentor,
                                              mode, loss_opt, loss_weight)

        self.setWeightSmallSeg()
        self.setWeightInstanceBd()

    def setWeightSmallSeg(self, thres=400, ratio=4, dilate=2):
        self.weight_small_size = thres # 2d threshold for small size
        self.weight_zratio = ratio # resolution ration between z and x/y
        self.weight_small_dilate = dilate # size of the border

    def setWeightInstanceBd(self, bd_dist=2):
        self.weight_instance_bd = bd_dist # filter size

    def __getitem__(self, index):
        vol_size = self.sample_input_size
        valid_mask = None

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            pos = self.get_pos_seed(vol_size, seed)
            # make labels index smaller. o/w uint32 and float32 are not the same for some values
            out_label = relabel(crop_volume(self.label[pos[0]], vol_size, pos[1:])).astype(np.float32)
            out_input = (crop_volume(self.input[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)
            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image':out_input, 'label':out_label}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'], augmented['label']
                out_input = out_input.astype(np.float32)
                out_label = out_label.astype(np.int16)
                #print(out_input.shape, out_label.shape) #debug
                #print(out_input.dtype, out_label.dtype) #debug

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = (crop_volume(self.input[pos[0]], vol_size, pos[1:])/255.0).astype(np.float32)
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:]).astype(np.float32)
            
        # Turn segmentation label into affinity in Pytorch Tensor
        if out_label is not None:
            # check for invalid region (-1)
            seg_bad = np.array([-1]).astype(out_label.dtype)[0]
            valid_mask = out_label!=seg_bad
            out_label[out_label==seg_bad] = 0
            # gt-widen process during data_io
            # replicate-pad the aff boundary
            if self.target_opt == 1:
                out_label_orig = out_label>0
            elif self.target_opt in [2,3]:
                out_label_orig = out_label.copy()
            out_label = seg_to_aff(out_label).astype(np.float32)
            out_label = torch.from_numpy(out_label)

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input)
        out_input = out_input.unsqueeze(0)
       
        do_weight = self.loss_opt in [0,1,4]# exist wBCE/wMSE
        if self.mode == 'train':
            extra = []
            if do_weight: 
                extra += [rebalance_binary_class(out_label)]

            if self.target_opt == 1: # find small seg region
                label_small = get_small_seg(out_label_orig, self.weight_small_size, self.weight_zratio)
                label_small_mask = binary_dilation(label_small, iterations=self.weight_small_dilate).astype(np.float32)
                label_small = torch.from_numpy((out_label_orig * label_small_mask).astype(np.float32))
                extra += [label_small]
                if do_weight:
                    extra += [rebalance_binary_class(label_small)]

            if self.target_opt in [2,3]: # find instance bd
                # 2: no bg-bd
                do_bg = False if self.target_opt==2 else True
                label_instance_bd = get_instance_bd(out_label_orig, self.weight_instance_bd, do_bg=do_bg).astype(np.float32)
                label_instance_bd = torch.from_numpy(label_instance_bd[None,:]) # add "channel" dim

                extra += [label_instance_bd]
                if do_weight:
                    extra += [rebalance_binary_class(label_instance_bd)]
         
            if len(extra) == 0:
                return pos, out_input, out_label
            else:
                return pos, out_input, out_label, extra

        else:
            return pos, out_input
