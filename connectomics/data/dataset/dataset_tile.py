from __future__ import print_function, division
import numpy as np
import json
import random

import torch
import torch.utils.data


from . import VolumeDataset
from ..utils import crop_volume, rebalance_binary_class,relabel,seg_widen_border
from ...io import tileToVolume 

class TileDataset(torch.utils.data.Dataset):
    """Pytorch dataset class for large-scale tile-based dataset.

    Args:
        volume_json: json file for input image.
        label_json: json file for label.
        sample_input_size (tuple, int): model input size.
        sample_label_size (tuple, int): model output size.
        sample_stride (tuple, int): stride size for sampling.
        augmentor: data augmentor.
        valid_mask: the binary mask of valid regions.
        mode (str): training or inference mode.
    """
    def __init__(self, chunk_num, chunk_num_ind, chunk_iter, chunk_stride,
                 volume_json, label_json=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 sample_invalid_thres = [0.5,0],
                 augmentor=None,
                 valid_mask=None, target_opt=['0'], loss_opt=['1'],
                 mode='train', label_erosion=0, pad_size=[0,0,0]):
        # TODO: merge mito/mitoskel, syn/synpolarity
        self.sample_input_size = sample_input_size
        self.sample_label_size = sample_label_size
        self.sample_stride = sample_stride
        self.sample_invalid_thres = sample_invalid_thres
        self.augmentor = augmentor
        self.target_opt = target_opt
        self.loss_opt = loss_opt
        self.mode = mode
        self.label_erosion = label_erosion
        self.pad_size = pad_size

        self.chunk_step = 1
        if chunk_stride: # if do stride, 50% overlap
            self.chunk_step = 2

        self.chunk_num = chunk_num
        if len(chunk_num_ind) == 0:
            self.chunk_num_ind = range(np.prod(chunk_num))
        else:
            self.chunk_num_ind = chunk_num_ind
        self.chunk_id_done = []

        self.json_volume = json.load(open(volume_json))
        self.json_label = json.load(open(label_json)) if (label_json is not None and label_json!='') else None 
        self.json_size = [self.json_volume['depth'],self.json_volume['height'],self.json_volume['width']]

        self.coord_m = np.array([0,self.json_volume['depth'],0,self.json_volume['height'],0,self.json_volume['width']],int)
        self.coord = np.zeros(6,int)



    def get_coord_name(self):
        return '-'.join([str(x) for x in self.coord])

    def updatechunk(self, do_load=True):
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
        
        x0,x1 = np.floor(np.array([xid,xid+self.chunk_step])/(self.chunk_num[2]+self.chunk_step-1)*self.json_size[2]).astype(int)
        y0,y1 = np.floor(np.array([yid,yid+self.chunk_step])/(self.chunk_num[1]+self.chunk_step-1)*self.json_size[1]).astype(int)
        z0,z1 = np.floor(np.array([zid,zid+self.chunk_step])/(self.chunk_num[0]+self.chunk_step-1)*self.json_size[0]).astype(int)

        self.coord = np.array([z0,z1,y0,y1,x0,x1],int)

        if do_load:
            self.loadchunk()

    def loadchunk(self):
        coord_p = self.coord+[-self.pad_size[0],self.pad_size[0],-self.pad_size[1],self.pad_size[1],-self.pad_size[2],self.pad_size[2]]
        print('load tile',self.coord)
        # keep it in uint8 to save memory
        volume = [tileToVolume(self.json_volume['image'], coord_p, self.coord_m,\
                             tile_sz=self.json_volume['tile_size'],tile_st=self.json_volume['tile_st'],
                              tile_ratio=self.json_volume['tile_ratio'])]
        label = None
        if self.json_label is not None: 
            dt={'uint8':np.uint8,'uint16':np.uint16,'uint32':np.uint32,'uint64':np.uint64}
            # float32 may misrepresent large uint32/uint64 numbers -> relabel to decrease the label index
            label = [relabel(tileToVolume(self.json_label['image'], coord_p, self.coord_m,\
                                 tile_sz=self.json_label['tile_size'],tile_st=self.json_label['tile_st'],
                                 tile_ratio=self.json_label['tile_ratio'], ndim=self.json_label['ndim'],
                                 dt=dt[self.json_label['dtype']], do_im=0), do_type=True)]
            if self.label_erosion != 0:
                label[0] = seg_widen_border(label[0], self.label_erosion)
        self.dataset = VolumeDataset(volume,label,
                              self.sample_input_size,
                              self.sample_label_size,
                              self.sample_stride,
                              self.sample_invalid_thres,
                              self.augmentor, self.target_opt, self.loss_opt,
                              self.mode)
