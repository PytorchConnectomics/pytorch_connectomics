from typing import Optional, List
from collections import OrderedDict
import numpy as np
import random

import torch
import torch.utils.data

from skimage.morphology import remove_small_objects, binary_dilation
from skimage.measure import label as label_cc

from connectomics.data.utils import *
from .utils import *

class VolumeDatasetCenterPatch(torch.utils.data.Dataset):
    """
    Dataset class for streaming the center patch of 3D segmentation masks (instance masks or
    synapse masks). The label volumes are always required for this class.

    Args:
        label (list, required): list of label volumes.
        volume (list, optional): list of image volumes. Default: None
        label_type (str): type of the annotation. Default: ``'syn'``
        sample_size (int): model input size (width of a square). Default: 128
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        iter_num (int): total number of training iterations (-1 for inference). Default: -1
        data_mean (float): mean of pixels for images normalized to (0,1). Default: 0.5
        data_std (float): standard deviation of pixels for images normalized to (0,1). Default: 0.5
        data_match_act (str): the data is normalized to match the range of an activation. Default: ``'none'``
    """
    background: int = 0  # background label index

    def __init__(self,
                 label: list,
                 volume: Optional[list] = None,
                 label_type: str = 'syn',
                 sample_size: int = 128,
                 mode: str = 'train',
                 iter_num: int = -1,
                 data_mean: float = 0.5,
                 data_std: float = 0.5,
                 data_match_act='none',
                 **kwargs):

        assert mode in ['train','test']
        self.mode = mode        
        assert label_type in ['seg', 'syn']
        self.label_type = label_type

        self.label = label # label is always required
        self.num_vols = len(self.label)
        self.volume_size = [np.array(x.shape) for x in self.label]

        self.sample_size = sample_size
        self.volume = volume

        self.all_patches = OrderedDict()
        if self.label_type == 'syn':
            for i in range(self.num_vols):
                syn, seg = self.process_syn(label[i])
                im = self.volume[i] if self.volume is not None else None
                data_dict = self.process_patch(syn, seg, im, self.sample_size)
                self.all_patches[i] = data_dict
        else:
            raise NotImplementedError

        # normalization
        self.data_mean = data_mean
        self.data_std = data_std
        self.data_match_act = data_match_act

        # self.all_patches is a dict of dict
        self.num_bbox_list = [len(x.keys()) for _, x in self.all_patches.items()]
        self.num_bbox_cumsum = np.cumsum([0] + self.num_bbox_list)
        self.num_bbox = sum(self.num_bbox_list)
        self.iter_num = max(iter_num, self.num_bbox) if self.mode == 'train' else self.num_bbox
        print('Total number of samples to be generated: ', self.iter_num)

    def __len__(self):
        # total number of possible samples
        return self.iter_num

    def __getitem__(self, index):
        if self.mode == 'train':
            vol_id = random.randrange(self.num_vols)
            box_id = random.randrange(self.num_bbox_list[vol_id])
        elif self.mode == 'test':
            vol_id = np.argmax(index < self.num_bbox_cumsum) - 1
            box_id = index - self.num_bbox_cumsum[vol_id]

        pos, img, seg = self.all_patches[vol_id][box_id]
        return pos, self.prepare_image(img), self.prepare_label(seg)

    def prepare_image(self, image: np.ndarray):
        image = (image / 255.0).astype(np.float32)
        image = np.expand_dims(image, 0) # channel dim for gray-scale images
        image = normalize_image(image, self.data_mean, self.data_std)   
        return image

    def prepare_label(self, label: np.ndarray):
        if self.label_type == "syn":
            output = np.zeros(label.shape, dtype=np.float32)
            pos = np.logical_and((label % 2) == 1, label > 0).astype(np.float32)
            neg = np.logical_and((label % 2) == 0, label > 0).astype(np.float32)
            output = output + pos - neg
            return np.expand_dims(output, 0)
        
        raise NotImplementedError

    def process_syn(self, mask: np.ndarray, small_thres: int =  16):
        indices = np.unique(mask)
        is_semantic = len(indices) == 3 and (indices==[0,1,2]).all()
        if not is_semantic: # already an instance-level segmentation
            syn, seg = mask, (mask.copy() + 1) // 2
            return syn, seg

        seg = binary_dilation(mask.copy() != 0)
        seg = label_cc(seg).astype(int)
        seg = seg * (mask.copy() != 0).astype(int)
        seg = remove_small_objects(seg, small_thres)

        c2 = (mask.copy() == 2).astype(int)
        c1 = (mask.copy() == 1).astype(int)

        syn_pos = np.clip((seg * 2 - 1), a_min=0, a_max=None) * c1
        syn_neg = (seg * 2) * c2
        syn = np.maximum(syn_pos, syn_neg)
        return syn, seg

    def process_patch(self, syn, seg, img = None, sz: int = 128):
        crop_size = int(sz * 1.415) # considering rotation 
        data_dict = OrderedDict()
        seg_idx = np.unique(seg)[1:] # ignore background
        bbox_dict = index2bbox(seg, seg_idx, iterative=False)

        for i, idx in enumerate(seg_idx):
            temp = (seg == idx) # binary mask of the current synapse
            bbox = bbox_dict[idx]

            # find the most centric slice that contains foreground
            temp_crop = crop_ND(temp, bbox, end_included=True)
            crop_mid = (temp_crop.shape[0]-1) // 2
            idx_t = np.where(np.any(temp_crop, axis=(1,2)))[0] # index of slices containing True values
            z_mid_relative = idx_t[np.argmin(np.abs(idx_t-crop_mid))]
            z_mid_total = z_mid_relative + bbox[0]

            temp_2d = temp[z_mid_total]
            bbox_2d = bbox_ND(temp_2d)    

            y1, y2 = adjust_bbox(bbox_2d[0], bbox_2d[1], crop_size)
            x1, x2 = adjust_bbox(bbox_2d[2], bbox_2d[3], crop_size)
            crop_2d = [y1, y2, x1, x2]

            cropped_syn, syn_bbox, padding = crop_pad_data(syn, z_mid_total, crop_2d, mask=temp, return_box=True)
            cropped_img = crop_pad_data(img, z_mid_total, crop_2d, pad_val=128) if img is not None \
                else np.zeros_like(cropped_syn).astype(np.uint8)

            # calculate and save the angle of rotation.
            angle, _ = calculate_rot(cropped_syn, return_overlap=False, mode='linear')

            img_dtype, syn_dtype = cropped_img.dtype, cropped_syn.dtype
            rotate_img_zmid, rotate_syn_zmid, angle = rotateIm_polarity(
                cropped_img.astype(np.float32), cropped_syn.astype(np.float32), -angle)
            rotate_img_zmid = center_crop(rotate_img_zmid.astype(img_dtype), sz)
            rotate_syn_zmid = center_crop(rotate_syn_zmid.astype(syn_dtype), sz)
            data_dict[i] = [bbox, rotate_img_zmid, rotate_syn_zmid]

        return data_dict
