from __future__ import print_function, division
from typing import Optional, List, Union
import numpy as np
import json
import random

import torch
import torch.utils.data

from . import VolumeDataset
from ..augmentation import Compose
from ..utils import relabel, tile2volume

TARGET_OPT_TYPE = List[str]
WEIGHT_OPT_TYPE = List[List[str]]
AUGMENTOR_TYPE = Optional[Compose]


class TileDataset(torch.utils.data.Dataset):
    r"""Dataset class for large-scale tile-based datasets. Large-scale volumetric datasets are usually stored as 
    individual tiles. Directly loading them as a single array for training and inference is infeasible. This 
    class reads the paths of the tiles and construct smaller chunks for processing.

    Args:
        chunk_num (list): volume spliting parameters in :math:`(z, y, x)` order. Default: :math:`[2, 2, 2]` 
        chunk_ind (list): predefined list of chunks. Default: `None`
        chunk_ind_split (list): rank and world_size for spliting chunk_ind in multi-processing. Default: `None`
        chunk_iter (int): number of iterations on each chunk. Default: -1
        chunk_stride (bool): allow overlap between chunks. Default: `True`
        volume_json (str): json file for input image. Default: ``'path/to/image'``
        label_json (str, optional): json file for label. Default: `None`
        valid_mask_json (str, optional): json file for valid mask. Default: `None`
        mode (str): ``'train'``, ``'val'`` or ``'test'``. Default: ``'train'``
        pad_size (list): padding parameters in :math:`(z, y, x)` order. Default: :math:`[0,0,0]`
    """

    def __init__(self,
                 chunk_num: List[int] = [2, 2, 2],
                 chunk_ind: Optional[list] = None,
                 chunk_ind_split: Optional[Union[List[int], str]] = None,
                 chunk_iter: int = -1,
                 chunk_stride: bool = True,
                 volume_json: str = 'path/to/image.json',
                 label_json: Optional[str] = None,
                 valid_mask_json: Optional[str] = None,
                 mode: str = 'train',
                 pad_size: List[int] = [0, 0, 0],
                 **kwargs):

        self.kwargs = kwargs
        self.mode = mode
        self.chunk_iter = chunk_iter
        self.pad_size = pad_size

        self.chunk_step = 1
        if chunk_stride and self.mode == 'train':  # 50% overlap between volumes during training
            self.chunk_step = 2

        self.chunk_num = chunk_num
        self.chunk_ind = self.get_chunk_ind(
            chunk_ind, chunk_ind_split)
        self.chunk_id_done = []

        self.json_volume = json.load(open(volume_json))
        self.json_label = json.load(open(label_json)) if (
            label_json is not None) else None
        self.json_valid = json.load(open(valid_mask_json)) if (
            valid_mask_json is not None) else None
        self.json_size = [self.json_volume['depth'],
                          self.json_volume['height'],
                          self.json_volume['width']]

        self.coord_m = np.array([0, self.json_volume['depth'],
                                 0, self.json_volume['height'],
                                 0, self.json_volume['width']], int)
        self.coord = np.zeros(6, int)

    def get_chunk_ind(self, chunk_ind, split_rule):
        if chunk_ind is None:
            chunk_ind = list(
                range(np.prod(self.chunk_num)))

        if split_rule is not None:
            if isinstance(split_rule, str):
                split_rule = split_rule.split('-')

            assert len(split_rule) == 2
            rank, world_size = split_rule
            rank, world_size = int(rank), int(world_size)
            x = len(chunk_ind) // world_size
            low, high = rank * x, (rank + 1) * x
            # Last split needs to cover remaining chunks.
            if rank == world_size - 1:
                high = len(chunk_ind)
            chunk_ind = chunk_ind[low:high]

        return chunk_ind

    def get_coord_name(self):
        r"""Return the filename suffix based on the chunk coordinates.
        """
        return '-'.join([str(x) for x in self.coord])

    def updatechunk(self, do_load=True):
        r"""Update the coordinates to a new chunk in the large volume.
        """
        if len(self.chunk_id_done) == len(self.chunk_ind):
            self.chunk_id_done = []
        id_rest = list(set(self.chunk_ind)-set(self.chunk_id_done))
        if self.mode == 'train':
            id_sample = id_rest[int(np.floor(random.random()*len(id_rest)))]
        elif self.mode == 'test':
            id_sample = id_rest[0]
        self.chunk_id_done += [id_sample]

        zid = float(id_sample//(self.chunk_num[1]*self.chunk_num[2]))
        yid = float((id_sample//self.chunk_num[2]) % (self.chunk_num[1]))
        xid = float(id_sample % self.chunk_num[2])

        x0, x1 = np.floor(np.array([xid, xid+self.chunk_step])/(
            self.chunk_num[2]+self.chunk_step-1)*self.json_size[2]).astype(int)
        y0, y1 = np.floor(np.array([yid, yid+self.chunk_step])/(
            self.chunk_num[1]+self.chunk_step-1)*self.json_size[1]).astype(int)
        z0, z1 = np.floor(np.array([zid, zid+self.chunk_step])/(
            self.chunk_num[0]+self.chunk_step-1)*self.json_size[0]).astype(int)

        self.coord = np.array([z0, z1, y0, y1, x0, x1], int)

        if do_load:
            self.loadchunk()

    def loadchunk(self):
        r"""Load the chunk based on current coordinates and construct a VolumeDataset for processing.
        """
        coord_p = self.coord + [-self.pad_size[0], self.pad_size[0],
                                -self.pad_size[1], self.pad_size[1],
                                -self.pad_size[2], self.pad_size[2]]
        print('load chunk: ', coord_p)
        # keep it in uint8 to save memory
        volume = [tile2volume(self.json_volume['image'], coord_p, self.coord_m,
                              tile_sz=self.json_volume['tile_size'], tile_st=self.json_volume['tile_st'],
                              tile_ratio=self.json_volume['tile_ratio'])]

        label = None
        if self.json_label is not None:
            dt = {'uint8': np.uint8, 'uint16': np.uint16,
                  'uint32': np.uint32, 'uint64': np.uint64}
            # float32 may misrepresent large uint32/uint64 numbers -> relabel to decrease the label index
            label = [relabel(tile2volume(self.json_label['image'], coord_p, self.coord_m,
                                         tile_sz=self.json_label['tile_size'], tile_st=self.json_label['tile_st'],
                                         tile_ratio=self.json_label['tile_ratio'], dt=dt[self.json_label['dtype']],
                                         do_im=0), do_type=True)]

        valid_mask = None
        if self.json_valid is not None:
            valid_mask = [tile2volume(self.json_valid['image'], coord_p, self.coord_m,
                                      tile_sz=self.json_valid['tile_size'], tile_st=self.json_valid['tile_st'],
                                      tile_ratio=self.json_valid['tile_ratio'])]

        self.dataset = VolumeDataset(volume, label, valid_mask,
                                     mode=self.mode,
                                     # specify chunk iteration number for training and -1 for inference
                                     iter_num=self.chunk_iter if self.mode == 'train' else -1,
                                     **self.kwargs)
