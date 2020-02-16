from __future__ import print_function, division
import numpy as np
import random

import torch
import torch.utils.data

from .dataset import BaseDataset
from ..utils import crop_volume, rebalance_binary_class

class BinaryDataset(BaseDataset):
    """Pytorch dataset class for binary segmentation.

    Args:
        volume: input image stacks.
        label: synapse masks.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor: data augmentor.
        valid_mask: the binary mask of valid regions.
        mode (str): training or inference mode.
    """
    def __init__(self,
                 volume, label=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 valid_mask=None,
                 mode='train', loss_opt=0, target_opt=0):
        super(BinaryDataset, self).__init__(volume,label,
                                          sample_input_size,
                                          sample_label_size,
                                          sample_stride,
                                          augmentor,
                                          mode, loss_opt, loss_weight)
        
        self.valid_mask = valid_mask
        if valid_mask is not None:
            self.valid_mask = np.float32(valid_mask)

    def __getitem__(self, index):
        vol_size = self.sample_input_size

        # Train Mode Specific Operations:
        if self.mode == 'train':
            # 2. get input volume
            seed = np.random.RandomState(index)
            # if elastic deformation: need different receptive field
            # change vol_size first
            if self.valid_mask is not None:
                while True: # reject sampling
                    pos = self.get_pos_seed(vol_size, seed)
                    out_valid = crop_volume(self.valid_mask[pos[0]], vol_size, pos[1:])
                    if np.sum(out_valid) / np.float32(np.prod(np.array(out_valid.shape))) > 0.8:
                        break
                out_label = (crop_volume(self.label[pos[0]], vol_size, pos[1:])>0).astype(np.float32)
                        
            else:
                while True: # reject sampling
                    pos = self.get_pos_seed(vol_size, seed)
                    out_label = (crop_volume(self.label[pos[0]], vol_size, pos[1:])>0).astype(np.float32)
                    break
                    """
                    # for partial label
                    if np.sum(out_label) > np.prod(vol_size)*0.6:
                        break
                    else:
                        if random.random() > 0.90:    
                            break       
                    """

            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:]).astype(np.float32)/255
            # 3. augmentation
            if self.augmentor is not None:  # augmentation
                data = {'image':out_input, 'label':out_label}
                augmented = self.augmentor(data, random_state=seed)
                out_input, out_label = augmented['image'].astype(np.float32), augmented['label'].astype(np.float32)

        # Test Mode Specific Operations:
        elif self.mode == 'test':
            # test mode
            pos = self.get_pos_test(index)
            out_input = crop_volume(self.input[pos[0]], vol_size, pos[1:])
            out_label = None if self.label is None else crop_volume(self.label[pos[0]], vol_size, pos[1:])
            
        if out_label is not None:
            out_label = torch.from_numpy(out_label)
            if len(out_label.size()) == 3:
                out_label = out_label.unsqueeze(0)

        # Turn input to Pytorch Tensor, unsqueeze once to include the channel dimension:
        out_input = torch.from_numpy(out_input)
        out_input = out_input.unsqueeze(0)

        if self.mode == 'train':
            # Rebalancing
            if self.weight_opt == -1: # no need for weight
                return pos, out_input, out_label

            if self.weight_opt == 0: # fg/bg balance re-weight
                weight_factor, weight = rebalance_binary_class(out_label)
                return pos, out_input, out_label, weight
            elif self.weight_opt == 1: # focus on small object
                raise NotImplementedError("Need to implement weight_opt = 1 !")
                weight_factor, weight = rebalance_binary_class(out_label)
                obj_small_mask = rebalance_small_object(out_label)
                return pos, out_input, out_label, weight

        else:
            return pos, out_input

