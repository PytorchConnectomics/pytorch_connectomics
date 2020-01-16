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
    def __init__(self, dataset_type, chunk_num, chunk_iter, chunk_stride,
                 volume_json, label_json=None,
                 sample_input_size=(8, 64, 64),
                 sample_label_size=None,
                 sample_stride=(1, 1, 1),
                 augmentor=None,
                 valid_mask=None,
                 mode='train'):
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
        self.dataset = DATASET_MAP[dataset_type]
        self.sample_intput_size = sample_input_size
        self.sample_label_size = sample_label_size
        self.sample_stride = sample_stride
        self.augmentor = augmentor
        self.mode = mode

        self.chunk_step = 1
        if chunk_stride: # if do stride, 50% overlap
            self.chunk_step = 2

        self.chunk_num = chunk_num
        self.chunk_num_total = np.prod(chunk_num)
        self.chunk_id_done = []

        self.json_volume = json.load(open(volume_json))
        self.json_label = json.load(open(label_json)) if (label_json is not None or label_json!='') else None 
        self.json_size = [self.json_volume['depth'],self.json_volume['height'],self.json_volume['width']]



    def updatechunk(self):
        if len(self.chunk_id_done)==self.chunk_num_total:
            self.chunk_id_done = []
        id_rest = list(set(range(self.chunk_num_total))-set(self.chunk_id_done))
        id_sample = id_rest[int(np.floor(np.random.random()*self.chunk_num_total))]
        self.chunk_id_done += [id_sample]

        zid = float(id_sample//(self.chunk_num[1]*self.chunk_num[2]))
        yid = float((id_sample//self.chunk_num[2])%(self.chunk_num[1]))
        xid = float(id_sample%self.chunk_num[2])
        
        x0,x1 = np.floor(np.array([xid,xid+self.chunk_step])/(self.chunk_num[2]+self.chunk_step-1)*self.json_size[2]).astype(int)
        y0,y1 = np.floor(np.array([yid,yid+self.chunk_step])/(self.chunk_num[1]+self.chunk_step-1)*self.json_size[1]).astype(int)
        z0,z1 = np.floor(np.array([zid,zid+self.chunk_step])/(self.chunk_num[0]+self.chunk_step-1)*self.json_size[0]).astype(int)

        import pdb; pdb.set_trace()
        volume = tileToVolume(self.json_volume['image'], x0, x1, y0, y1, z0, z1,\
                             tile_sz=self.json_volume['tile_size'],tile_st=self.json_volume['tile_st'],
                             tile_ratio=self.json_volume['tile_ratio'])
        import pdb; pdb.set_trace()
        label = None
        if self.json_label is not None: 
            label = tileToVolume(self.json_label['image'], x0, x1, y0, y1, z0, z1,\
                                 tile_sz=self.json_label['tile_size'],tile_st=self.json_label['tile_st'],
                                 tile_ratio=self.json_label['tile_ratio'], ndim=self.json_label['ndim'], resize_order=0)
            import pdb; pdb.set_trace()

        self.dataset.__init__(volume,
                              label,
                              self.sample_input_size,
                              self.sample_label_size,
                              self.sample_stride,
                              self.augmentor,
                              self.mode)
