from __future__ import print_function, division
from typing import Optional, List, Union
import numpy as np
import json
import random

import torch
import torch.utils.data
from scipy.ndimage import zoom

from . import VolumeDataset
from ..utils import reduce_label, tile2volume


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
        pad_size (list): padding parameters in :math:`(z, y, x)` order. Default: :math:`[0, 0, 0]`
        data_scale (list): volume scaling factors in :math:`(z, y, x)` order. Default: :math:`[1.0, 1.0, 1.0]`
        coord_range (list): the valid coordinate range of volumes. Default: `None`
        do_relabel (bool): reduce the the mask indicies in a sampled label volume. This option be set to
            False for semantic segmentation, otherwise the classes can shift. Default: True

    Note:
        To run inference using multiple nodes in an asynchronous manner, ``chunk_ind_split`` specifies the number of
        parts to split the total number of chunks in inference, and which part should the current node/process see. For
        example, ``chunk_ind_split = "0-5"`` means the chunks are split into 5 parts (thus can be processed asynchronously
        using 5 nodes), and the current node/process is handling the first (0-base) part of the chunks.

    Note:
        The ``coord_range`` option specify the region of a volume to use. Suppose the fisrt input volume has a voxel size 
        of (1000, 10000, 10000), and only the center subvolume of size (400, 2000, 2000) needs to be used for training or 
        inference, then set ``coord_range=[[300, 700, 4000, 6000, 4000, 6000]]``.
    """

    def __init__(self,
                 chunk_num: List[int] = [2, 2, 2],
                 chunk_ind: Optional[list] = None,
                 chunk_ind_split: Optional[Union[List[int], str]] = None,
                 chunk_iter: int = -1,
                 chunk_stride: bool = True,
                 volume_json: List[str] = ['path/to/image.json'],
                 label_json: Optional[List[str]] = None,
                 valid_mask_json: Optional[List[str]] = None,
                 mode: str = 'train',
                 pad_size: List[int] = [0, 0, 0],
                 data_scale: List[float] = [1.0, 1.0, 1.0],
                 coord_range: Optional[List[List[int]]] = None,
                 do_relabel: bool = True,
                 **kwargs):

        self.kwargs = kwargs
        self.mode = mode
        self.chunk_iter = chunk_iter
        self.pad_size = pad_size
        self.data_scale = data_scale
        self.do_relabel = do_relabel

        self.chunk_step = 1
        if chunk_stride and self.mode == 'train':  # 50% overlap between volumes during training
            self.chunk_step = 2

        self.chunk_num = chunk_num
        self.chunk_ind = self.get_chunk_ind(chunk_ind, chunk_ind_split)
        self.chunk_id_done = []

        self.num_volumes = len(volume_json) # number of volumes specified by json files
        if self.mode == 'test':
            assert self.num_volumes == 1, "Only one json file should be given in inference!"

        self.json_volume = [json.load(open(volume_json[i])) for i in range(self.num_volumes)]
        self.json_label = [json.load(open(label_json[i])) for i in range(self.num_volumes)] if (
            label_json is not None) else None
        self.json_valid = [json.load(open(valid_mask_json[i])) for i in range(self.num_volumes)] if (
            valid_mask_json is not None) else None

        self.json_size = [[
            self.json_volume[i]['depth'],
            self.json_volume[i]['height'],
            self.json_volume[i]['width']]
            for i in range(self.num_volumes)] 

        self.coord_m = np.array([[
            0, self.json_volume[i]['depth'],
            0, self.json_volume[i]['height'],
            0, self.json_volume[i]['width']]
            for i in range(self.num_volumes)], int)

        # specify the coordintate range of data to use
        self.get_coord_range(coord_range)

    def get_chunk_ind(self, chunk_ind, split_rule):
        if chunk_ind is None:
            chunk_ind = list(
                range(np.prod(self.chunk_num)))

        if split_rule is not None:
            # keep only the chunk indicies for current node/process
            if isinstance(split_rule, str):
                split_rule = split_rule.split('-')

            assert len(split_rule) == 2
            rank, world_size = split_rule
            rank, world_size = int(rank), int(world_size)
            assert rank < world_size # rank is 0-base
            x = len(chunk_ind) // world_size
            low, high = rank * x, (rank + 1) * x
            # Last split needs to cover remaining chunks.
            if rank == world_size - 1:
                high = len(chunk_ind)
            chunk_ind = chunk_ind[low: high]

        return chunk_ind

    def get_coord_name(self):
        r"""Return the filename suffix based on the chunk coordinates.
        """
        # this function should only be called in test mode
        assert self.mode == 'test' and len(self.coord) == 1
        return '-'.join([str(x) for x in self.coord[0]])

    def get_coord_range(self, coord_range):
        if coord_range is not None:
            if isinstance(coord_range[0], int):
                assert len(coord_range) == 6
                self.coord_range = [coord_range for _ in range(self.num_volumes)]
            elif isinstance(coord_range[0], list):
                self.coord_range = coord_range
        else:
            self.coord_range = self.coord_m # use all data

        assert len(self.coord_range) == self.num_volumes
        self.coord_range_l, self.coord_range_r = [], []
        for temp in self.coord_range: # lower and higher boundaries
            self.coord_range_l.append([temp[2*i] for i in range(3)])
            self.coord_range_r.append([temp[2*i+1] for i in range(3)])

    def get_range_axis(self, axis_id, vol_id, axis: str='z'):
        axis_map = {'z': 0, 'y': 1, 'x': 2}
        l_bd = self.coord_range_l[vol_id][axis_map[axis]]
        r_bd = self.coord_range_r[vol_id][axis_map[axis]]
        assert r_bd > l_bd
        length = r_bd - l_bd

        steps = np.array([axis_id, axis_id + self.chunk_step])
        axis_range = np.floor(steps / (
            self.chunk_num[axis_map[axis]] + self.chunk_step-1) * length).astype(int)
        return axis_range + l_bd

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

        self.coord = []
        for i in range(self.num_volumes):
            z0, z1 = self.get_range_axis(zid, vol_id=i, axis='z')
            y0, y1 = self.get_range_axis(yid, vol_id=i, axis='y')
            x0, x1 = self.get_range_axis(xid, vol_id=i, axis='x')
            self.coord.append(np.array([z0, z1, y0, y1, x0, x1], int))

        if do_load:
            self.loadchunk()

    def loadchunk(self):
        r"""Load the chunk based on current coordinates and construct a VolumeDataset for processing.
        """
        # assuming same padding for the list of volumes given
        padding = np.array([
            -self.pad_size[0], self.pad_size[0],
            -self.pad_size[1], self.pad_size[1],
            -self.pad_size[2], self.pad_size[2]])

        coord_p = [self.coord[i] + padding for i in range(self.num_volumes)]
        print('load chunk: ', coord_p)

        volume = [
            tile2volume(self.json_volume[i]['image'], coord_p[i], self.coord_m[i],
            tile_sz=self.json_volume[i]['tile_size'], tile_st=self.json_volume[i]['tile_st'],
            tile_ratio=self.json_volume[i]['tile_ratio']) for i in range(self.num_volumes)
        ]
        volume = self.maybe_scale(volume, order=3)

        label = None
        if self.json_label is not None:
            dt = {'uint8': np.uint8, 'uint16': np.uint16, 'uint32': np.uint32, 'uint64': np.uint64}
            label = [
                tile2volume(self.json_label[i]['image'], coord_p[i], self.coord_m[i],
                            tile_sz=self.json_label[i]['tile_size'], tile_st=self.json_label[i]['tile_st'],
                            tile_ratio=self.json_label[i]['tile_ratio'], dt=dt[self.json_label[i]['dtype']],
                            do_im=False) for i in range(self.num_volumes)
            ]
            # float32 may misrepresent large uint32/uint64 numbers -> reduce the label indices
            if self.do_relabel:
                label = [reduce_label(x, do_type=True) for x in label]
            label = self.maybe_scale(label, order=0)

        valid_mask = None
        if self.json_valid is not None:
            valid_mask = [
                tile2volume(self.json_valid[i]['image'], coord_p[i], self.coord_m[i],
                            tile_sz=self.json_valid[i]['tile_size'], tile_st=self.json_valid[i]['tile_st'],
                            tile_ratio=self.json_valid[i]['tile_ratio'], do_im=False) for i in range(self.num_volumes)
            ]
            valid_mask = self.maybe_scale(valid_mask, order=0)

        self.dataset = VolumeDataset(volume, label, valid_mask, mode=self.mode, do_relabel=self.do_relabel,
                                     # specify chunk iteration number for training (-1 for inference)
                                     iter_num=self.chunk_iter if self.mode == 'train' else -1,
                                     **self.kwargs)

    def maybe_scale(self, data, order=0):
        if (np.array(self.data_scale) != 1).any():
            for i in range(len(data)):
                dt = data[i].dtype
                data[i] = zoom(data[i], self.data_scale,
                               order=order).astype(dt)

        return data
