from __future__ import print_function, division
from typing import Optional, List
import numpy as np
import json
import random

import torch
import torch.utils.data

from . import VolumeDataset
from ..augmentation import Compose
from ..utils import crop_volume, relabel,seg_widen_border, tileToVolume 

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]

class TileDataset(torch.utils.data.Dataset):
    r"""Dataset class for large-scale tile-based datasets. Large-scale volumetric datasets are usually stored as 
    individual tiles. Directly loading them as a single array for training and inference is infeasible. This 
    class reads the paths of the tiles and construct smaller chunks for processing.

    Args:
        chunk_num (list): volume spliting parameters in :math:`(z, y, x)` order. Default: :math:`[2, 2, 2]` 
        chunk_num_ind (list): predefined list of chunks. Default: `None`
        chunk_iter (int): number of iterations on each chunk. Default: -1
        chunk_stride (bool): allow overlap between chunks. Default: `True`
        volume_json (str): json file for input image. Default: ``'path/to/image'``
        label_json (str, optional): json file for label. Default: `None`
        valid_mask_json (str, optional): json file for valid mask. Default: `None`
        valid_ratio (float): volume ratio threshold for valid samples. Default: 0.5
        sample_volume_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor (connectomics.data.augmentation.composition.Compose, optional): data augmentor for training. Default: `None`
        target_opt (list): list the model targets generated from segmentation labels.
        weight_opt (list): list of options for generating pixel-wise weight masks.
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        do_2d (bool): load 2d samples from 3d volumes. Default: `False`
        label_erosion (int): label erosion parameter to widen border. Default: 0
        pad_size(list): padding parameters in :math:`(z, y, x)` order. Default: :math:`[0,0,0]`
        reject_size_thres (int): threshold to decide if a sampled volumes contains foreground objects. Default: 0
        reject_p (float): probability of rejecting non-foreground volumes. Default: 0.95
    """

    def __init__(self, 
                 chunk_num: List[int] = [2, 2, 2], 
                 chunk_num_ind: Optional[list] = None,
                 chunk_iter: int = -1, 
                 chunk_stride: bool = True,
                 volume_json: str = 'path/to/image', 
                 label_json: Optional[str] = None,
                 valid_mask_json: Optional[str] = None,
                 valid_ratio: float = 0.5,
                 sample_volume_size: tuple = (8, 64, 64),
                 sample_label_size: Optional[tuple] = None,
                 sample_stride: tuple = (1, 1, 1),
                 augmentor: AUGMENTOR_TYPE = None,
                 target_opt: TARGET_OPT_TYPE = ['0'],
                 weight_opt: WEIGHT_OPT_TYPE = [['1']],
                 mode: str = 'train', 
                 do_2d: bool = False,
                 label_erosion: int = 0, 
                 pad_size: List[int] = [0,0,0],
                 reject_size_thres: int = 0,
                 reject_p: float = 0.95):
        
        self.sample_volume_size = sample_volume_size
        self.sample_label_size = sample_label_size
        self.sample_stride = sample_stride
        self.valid_ratio = valid_ratio
        self.augmentor = augmentor

        self.target_opt = target_opt
        self.weight_opt = weight_opt

        self.mode = mode
        self.do_2d = do_2d
        self.chunk_iter = chunk_iter
        self.label_erosion = label_erosion
        self.pad_size = pad_size

        self.chunk_step = 1
        if chunk_stride: # if do stride, 50% overlap
            self.chunk_step = 2

        self.chunk_num = chunk_num
        if chunk_num_ind is None:
            self.chunk_num_ind = range(np.prod(chunk_num))
        else:
            self.chunk_num_ind = chunk_num_ind
        self.chunk_id_done = []

        self.json_volume = json.load(open(volume_json))
        self.json_label = json.load(open(label_json)) if (label_json is not None) else None
        self.json_valid = json.load(open(valid_mask_json)) if (valid_mask_json is not None) else None
        self.json_size = [self.json_volume['depth'],
                          self.json_volume['height'],
                          self.json_volume['width']]

        self.coord_m = np.array([0, self.json_volume['depth'],
                                 0, self.json_volume['height'],
                                 0, self.json_volume['width']], int)
        self.coord = np.zeros(6, int)

        # rejection samping
        self.reject_size_thres = reject_size_thres
        self.reject_p = reject_p        

    def get_coord_name(self):
        r"""Return the filename suffix based on the chunk coordinates.
        """
        return '-'.join([str(x) for x in self.coord])

    def updatechunk(self, do_load=True):
        r"""Update the coordinates to a new chunk in the large volume.
        """
        if len(self.chunk_id_done)==len(self.chunk_num_ind):
            self.chunk_id_done = []
        id_rest = list(set(self.chunk_num_ind)-set(self.chunk_id_done))
        if self.mode == 'train':
            id_sample = id_rest[int(np.floor(random.random()*len(id_rest)))]
        elif self.mode == 'test':
            id_sample = id_rest[0]
        self.chunk_id_done += [id_sample]

        zid = float(id_sample//(self.chunk_num[1]*self.chunk_num[2]))
        yid = float((id_sample//self.chunk_num[2])%(self.chunk_num[1]))
        xid = float(id_sample%self.chunk_num[2])
        
        x0, x1 = np.floor(np.array([xid,xid+self.chunk_step])/(self.chunk_num[2]+self.chunk_step-1)*self.json_size[2]).astype(int)
        y0, y1 = np.floor(np.array([yid,yid+self.chunk_step])/(self.chunk_num[1]+self.chunk_step-1)*self.json_size[1]).astype(int)
        z0, z1 = np.floor(np.array([zid,zid+self.chunk_step])/(self.chunk_num[0]+self.chunk_step-1)*self.json_size[0]).astype(int)

        self.coord = np.array([z0, z1, y0, y1, x0, x1], int)

        if do_load:
            self.loadchunk()

    def loadchunk(self):
        r"""Load the chunk based on current coordinates and construct a VolumeDataset for processing.
        """
        coord_p = self.coord+[-self.pad_size[0],self.pad_size[0],-self.pad_size[1],self.pad_size[1],-self.pad_size[2],self.pad_size[2]]
        print('load tile', self.coord)
        # keep it in uint8 to save memory
        volume = [tileToVolume(self.json_volume['image'], coord_p, self.coord_m, \
                               tile_sz=self.json_volume['tile_size'], tile_st=self.json_volume['tile_st'],
                               tile_ratio=self.json_volume['tile_ratio'])]
                              
        label = None
        if self.json_label is not None: 
            dt={'uint8':np.uint8, 'uint16':np.uint16, 'uint32':np.uint32, 'uint64':np.uint64}
            # float32 may misrepresent large uint32/uint64 numbers -> relabel to decrease the label index
            label = [relabel(tileToVolume(self.json_label['image'], coord_p, self.coord_m, \
                                 tile_sz=self.json_label['tile_size'],tile_st=self.json_label['tile_st'],
                                 tile_ratio=self.json_label['tile_ratio'], ndim=self.json_label['ndim'],
                                 dt=dt[self.json_label['dtype']], do_im=0), do_type=True)]
            if self.label_erosion != 0:
                label[0] = seg_widen_border(label[0], self.label_erosion)

        valid_mask = None
        if self.json_valid is not None:
            valid_mask = [tileToVolume(self.json_valid['image'], coord_p, self.coord_m, \
                          tile_sz=self.json_valid['tile_size'], tile_st=self.json_valid['tile_st'],
                          tile_ratio=self.json_valid['tile_ratio'])]
                
        self.dataset = VolumeDataset(volume, label, valid_mask,
                valid_ratio = self.valid_ratio,
                sample_volume_size = self.sample_volume_size,
                sample_label_size = self.sample_label_size,
                sample_stride = self.sample_stride,
                augmentor = self.augmentor,
                target_opt = self.target_opt,
                weight_opt = self.weight_opt,
                mode = self.mode,
                do_2d = self.do_2d,
                iter_num = self.chunk_iter,
                reject_size_thres = self.reject_size_thres,
                reject_p = self.reject_p)
