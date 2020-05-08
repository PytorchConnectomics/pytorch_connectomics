import numpy as np
import itertools
import torch

class TestAugmentor(object):
    """Test Augmentor.

    Args:
        mode (str): inference mode ('min', 'max', 'mean').
        num_aug (int): number of data augmentations: 4-fold, 16-fold
    """
    def __init__(self, mode='min', num_aug=4):
        self.mode = mode
        self.num_aug = num_aug

    def __call__(self, model, data):
        out = None
        cc = 0
        if self.num_aug ==4:
            opts = itertools.product((False, ), (False, True), (False, True), (False, ))
        else:
            opts = itertools.product((False, True), (False, True), (False, True), (False, True))

        for xflip, yflip, zflip, transpose in opts:
            extension = ""
            if transpose:
                extension += "t"
            if zflip:
                extension += "z"
            if yflip:
                extension += "y"
            if xflip:
                extension += "x"
            volume = data.clone()
            # batch_size,channel,z,y,x 

            if xflip:
                volume = torch.flip(volume, [4])
            if yflip:
                volume = torch.flip(volume, [3])
            if zflip:
                volume = torch.flip(volume, [2])
            if transpose:
                volume = torch.transpose(volume, 3, 4)
            # aff: 3*z*y*x 
            vout = model(volume).cpu().detach().numpy()

            if transpose: # swap x-/y-affinity
                vout = vout.transpose(0, 1, 2, 4, 3)
                vout[:,[1,2]] = vout[:,[2,1]]
            if zflip:
                vout = vout[:,:,::-1]
            if yflip:
                vout = vout[:,:, :, ::-1]
            if xflip:
                vout = vout[:,:, :, :, ::-1]
                
            if out is None:
                if self.mode == 'min':
                    out = np.ones(vout.shape,dtype=np.float32)
                elif self.mode == 'max':
                    out = np.zeros(vout.shape,dtype=np.float32)
                elif self.mode == 'mean':
                    out = np.zeros(vout.shape,dtype=np.float32)

            if self.mode == 'min':
                out = np.minimum(out, vout)
            elif self.mode == 'max':
                out = np.maximum(out, vout)
            elif self.mode == 'mean':
                out += vout
            cc+=1

        if self.mode == 'mean':
            out = out/cc

        return out
