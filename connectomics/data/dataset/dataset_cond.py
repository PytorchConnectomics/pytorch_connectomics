from typing import Optional, List
import numpy as np
import random

import torch
import torch.utils.data
from skimage.morphology import remove_small_objects

from ..augmentation import Compose
from ..utils import *

AUGMENTOR_TYPE = Optional[Compose]
WEIGHT_OPT_TYPE = List[List[str]]


class VolumeDatasetCond(torch.utils.data.Dataset):
    """
    Dataset class for volumetric images in conditional segmentation. The label volumes are always required for this class.

    Args:
        label (list): list of label volumes.
        volume (list): list of image volumes.
        label_type (str): type of the annotation. Default: ``'syn'``
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: None
        sample_size (tuple): model input size. Default: (9, 65, 65)
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        data_mean (float): mean of pixels for images normalized to (0,1). Default: 0.5
        data_std (float): standard deviation of pixels for images normalized to (0,1). Default: 0.5      
    """
    background: int = 0  # background label index
    bbox_iterative: bool = False # calculate bbox iteratively (can be very slow if True)

    def __init__(self,
                 volume: List[np.ndarray],
                 label: List[np.ndarray],
                 label_type: str = 'syn',
                 augmentor: AUGMENTOR_TYPE = None,
                 sample_size: tuple = (9, 65, 65),
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 mode: str = 'train',
                 iter_num: int = -1,
                 data_mean: float = 0.5,
                 data_std: float = 0.5):
        assert mode in ['train','test']
        self.mode = mode        
        assert label_type in ['seg', 'syn']
        self.label_type = label_type
        self.augmentor = augmentor

        self.volume = volume 
        self.num_vols = len(self.volume)
        self.weight_opt = weight_opt

        self.volume_size = [np.array(x.shape) for x in self.volume]
        self.sample_size = sample_size
        self.label = label # label is always required in conditional segmentation
    
        if self.label_type == 'syn':
            self.aux_label = [(x+1) // 2 for x in self.label] # merge pre/post synaptic masks
            self.bbox_dict, self.idx_dict = self.get_bounding_box(self.aux_label)
                            
            # convert the 2D bounding boxes (z0=z1) to 3D
            if self.mode == 'test':
                swell_z = (self.sample_size[0]-1)//2

                for i in range(self.num_vols):
                    zl, zh = 0, self.volume_size[i][0]
                    for l, bb in enumerate(self.bbox_dict[i]):
                        z0, z1, y0, y1, x0, x1 = bb
                        z0 = max(z0-swell_z, zl)
                        z1 = min(z1+swell_z, zh)
                        self.bbox_dict[i][l] = (z0, z1, y0, y1, x0, x1)

        else: # each mask index is associated with one segment
            self.bbox_dict, self.idx_dict = self.get_bounding_box(self.label)

        # normalization
        self.data_mean = data_mean
        self.data_std = data_std

        self.num_bbox = sum([len(x) for _, x in self.bbox_dict.items()])
        self.iter_num = max(iter_num, self.num_bbox) if self.mode == 'train' else self.num_bbox
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        if self.mode == 'train':
            # randomly sample a volume
            vol_id = random.randrange(self.num_vols)
            bbox_list = self.bbox_dict[vol_id]
            box_id = random.randrange(len(bbox_list))
            bbox = bbox_list[box_id]
            crop_box = self.update_box(bbox)

            pos = (vol_id,) + tuple(bbox)
            out_volume = self.prepare_volume(crop_box, vol_id)
            out_target = self.prepare_label(crop_box, vol_id, box_id)
            out_weight = seg_to_weights(out_target, self.weight_opt)
            return pos, out_volume, out_target, out_weight

        elif self.mode == 'test':
            
            # ToDo: Inferencing can currently only handle a single volume
            assert (self.num_vols == 1), "Only provide a single volume for testing"

            bbox_list = self.bbox_dict[0]
            bbox = bbox_list[index]

            for i in range(len(bbox)//2):
                # assert upper bound is larger then lower bound
                assert (bbox[(i*2)+1] >= bbox[i*2])
                # truncate bbox if larger then model sample_size
                if bbox[(i*2)+1]-bbox[i*2] > self.sample_size[i]:
                    bbox = list(bbox)
                    print(f"Truncating the bounding box {index} by {bbox[(i*2)+1] - (bbox[i*2] + self.sample_size[i]-1)} for axes {i}.")
                    bbox[(i*2)+1] = bbox[i*2] + self.sample_size[i]-1
                    bbox = tuple(bbox)

            crop_box = self.update_box(bbox)
            pos = (0,) + tuple(bbox)
            out_volume = self.prepare_volume(crop_box, 0)
            out_target = self.prepare_label(crop_box, 0, index)

            # return the original pos and the pos of the cropped/padded box
            return [pos, (0,) + tuple(crop_box)], out_volume, out_target


    def prepare_volume(self, box, vol_id):
        image = self.crop_with_box(box, self.volume[vol_id], constant_values=128)
        image = (image / 255.0).astype(np.float32)
        image = np.expand_dims(image, 0) # channel dim for gray-scale images
        image = normalize_image(image, self.data_mean, self.data_std)   
        return image

    def prepare_label(self, box, vol_id, box_id):
        idx = self.idx_dict[vol_id][box_id]
        # target is a list to be consistent with VolumeDataset
        label = self.crop_with_box(box, self.label[vol_id])  
        if self.label_type == 'seg':
            return [(label==idx).astype(np.float32)]

        # synaptic polarity
        assert self.label_type == 'syn'
        gating_mask = self.crop_with_box(box, self.aux_label[vol_id])
        gating_mask = (gating_mask==idx).astype(label.dtype)
        label = label * gating_mask
        return [seg2polarity(label)]

    def crop_with_box(self, box, vol, constant_values = 0):
        # crop with given box (needs padding if touch boundary)
        assert len(box) == 6
        z0, z1, y0, y1, x0, x1 = box
        zh, yh, xh = vol.shape
        zl, yl, xl = [0, 0, 0]

        boundary = np.array([z0<zl, z1>zh, y0<yl, y1>yh, x0<xl, x1>xh]).astype(int)
        pad_size = np.array([z0-zl, z1-zh, y0-yl, y1-yh, x0-xl, x1-xh])
        pad_size = get_padsize(boundary*np.abs(pad_size), ndim=3)
        
        z0, y0, x0 = max(z0, zl), max(y0, yl), max(x0, xl)
        z1, y1, x1 = min(z1, zh), min(y1, yh), min(x1, xh)
        temp = vol[z0:z1, y0:y1, x0:x1]
        return np.pad(temp, pad_size, mode='constant', 
                      constant_values=constant_values)

    def update_box(self, bbox):
        z0, z1 = rand_window(bbox[0], bbox[1]+1, self.sample_size[0])
        y0, y1 = rand_window(bbox[2], bbox[3]+1, self.sample_size[1])
        x0, x1 = rand_window(bbox[4], bbox[5]+1, self.sample_size[2])
        return [z0, z1, y0, y1, x0, x1]        

    def get_bounding_box(self, label: list):
        # calculate the bounding box
        bbox_dict, idx_dict = {}, {}
        for i, vol in enumerate(label):
            bbox_dict[i], idx_dict[i] = self.get_bbox_vol(vol)
        return bbox_dict, idx_dict

    def get_bbox_vol(self, vol: np.ndarray, min_size: int=128):     
        vol = remove_small_objects(vol, min_size) # remove tiny objects

        indices = np.unique(vol)
        assert indices[0] == 0 # background
        indices = indices[1:]
        ind2box_dict = index2bbox(vol, indices, iterative=self.bbox_iterative)
        bbox_list = list(ind2box_dict.values())
        idx_list = list(ind2box_dict.keys())
        return bbox_list, idx_list
