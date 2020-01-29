from __future__ import print_function, division
import numpy as np
import json

import torch
import torch.utils.data

from .dataset import BaseDataset
from .misc import crop_volume, rebalance_binary_class   

from torch_connectomics.data.utils.functional_transform import skeleton_transform_volume
from torch_connectomics.data.dataset import *
from torch_connectomics.data.dataset.misc import tileToVolume
from torch_connectomics.libs.seg.seg_util import relabel,widen_border3

class TileDataset(BaseDataset):
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
    def __init__(self, dataset_type, chunk_num, chunk_num_ind, chunk_iter, chunk_stride,
                 volume_json, label_json=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 valid_mask=None,
                 mode='train', weight_opt=0, label_erosion=0, pad_size=[0,0,0]):
        # TODO: merge mito/mitoskel, syn/synpolarity
        DATASET_MAP = {'mito': MitoDataset,
                      'mitoskel': MitoSkeletonDataset,
                     'synapse': SynapseDataset,
                     'synapsepolarity': SynapsePolarityDataset,
                     'affinity': AffinityDataset,
                      }
        DATASET_MAP = {2: MitoDataset,
                      22: MitoSkeletonDataset,
                      1: SynapseDataset,
                      11: SynapsePolarityDataset,
                      0: AffinityDataset,
                      }
        self.dataset_type = DATASET_MAP[dataset_type]
        self.sample_input_size = sample_input_size
        self.sample_label_size = sample_label_size
        self.sample_stride = sample_stride
        self.augmentor = augmentor
        self.weight_opt = weight_opt
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



    def updatechunk(self):
        if len(self.chunk_id_done)==len(self.chunk_num_ind):
            self.chunk_id_done = []
        id_rest = list(set(self.chunk_num_ind)-set(self.chunk_id_done))
        if self.mode == 'train':
            id_sample = id_rest[int(np.floor(np.random.random()*len(id_rest)))]
        elif self.mode == 'test':
            id_sample = id_rest[0]
        self.chunk_id_done += [id_sample]

        zid = float(id_sample//(self.chunk_num[1]*self.chunk_num[2]))
        yid = float((id_sample//self.chunk_num[2])%(self.chunk_num[1]))
        xid = float(id_sample%self.chunk_num[2])
        
        x0,x1 = np.floor(np.array([xid,xid+self.chunk_step])/(self.chunk_num[2]+self.chunk_step-1)*self.json_size[2]).astype(int)
        y0,y1 = np.floor(np.array([yid,yid+self.chunk_step])/(self.chunk_num[1]+self.chunk_step-1)*self.json_size[1]).astype(int)
        z0,z1 = np.floor(np.array([zid,zid+self.chunk_step])/(self.chunk_num[0]+self.chunk_step-1)*self.json_size[0]).astype(int)


        if self.mode=='test': # padding or not
            coord = np.array([z0,z1,y0,y1,x0,x1],int)
            z0 = max(0, z0-self.pad_size[0])
            y0 = max(0, y0-self.pad_size[1])
            x0 = max(0, x0-self.pad_size[2])
            z1 = min(z1+self.pad_size[0], self.json_volume['depth'])
            y1 = min(y1+self.pad_size[1], self.json_volume['height'])
            x1 = min(x1+self.pad_size[2], self.json_volume['width'])
        # keep it in uint8 to save memory
        volume = [tileToVolume(self.json_volume['image'], x0, x1, y0, y1, z0, z1,\
                             tile_sz=self.json_volume['tile_size'],tile_st=self.json_volume['tile_st'],
                              tile_ratio=self.json_volume['tile_ratio'])]
        if self.mode=='test': # padding or not
            to_pad = [(0,0),(0,0),(0,0)]
            to_pad[0] = (self.pad_size[0]-(coord[0]-z0), self.pad_size[0]-(z1-coord[1]))
            to_pad[1] = (self.pad_size[1]-(coord[2]-y0), self.pad_size[1]-(y1-coord[3]))
            to_pad[2] = (self.pad_size[2]-(coord[4]-x0), self.pad_size[2]-(x1-coord[5]))
            print('test:', to_pad)
            if max(to_pad[0])+max(to_pad[1])+max(to_pad[2]) > 0:
                volume[0] = np.pad(volume[0],to_pad,mode='reflect')
            
        label = None
        if self.json_label is not None: 
            dt={'uint8':np.uint8,'uint16':np.uint16,'uint32':np.uint32,'uint64':np.uint64}
            # float32 may misrepresent large uint32/uint64 numbers -> relabel to decrease the label index
            label = [relabel(tileToVolume(self.json_label['image'], x0, x1, y0, y1, z0, z1,\
                                 tile_sz=self.json_label['tile_size'],tile_st=self.json_label['tile_st'],
                                 tile_ratio=self.json_label['tile_ratio'], ndim=self.json_label['ndim'],
                                 dt=dt[self.json_label['dtype']], resize_order=0), do_type=True)]
            if self.label_erosion != 0:
                label[0] = widen_border3(label[0], self.label_erosion)
        self.dataset = self.dataset_type(volume,label,
                              self.sample_input_size,
                              self.sample_label_size,
                              self.sample_stride,
                              self.augmentor,
                              self.mode, self.weight_opt)
