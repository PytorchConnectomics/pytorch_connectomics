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
        assert num_aug in [4, 16], "TestAugmentor.num_aug should be either 4 or 16!"

    def __call__(self, model, data):
        out = None
        cc = 0
        if self.num_aug == 4:
            opts = itertools.product((False, ), (False, ), (False, True), (False, True))
        else:
            opts = itertools.product((False, True), (False, True), (False, True), (False, True))

        for xflip, yflip, zflip, transpose in opts:
            volume = data.clone()
            # b,c,z,y,x 

            if xflip:
                volume = torch.flip(volume, [4])
            if yflip:
                volume = torch.flip(volume, [3])
            if zflip:
                volume = torch.flip(volume, [2])
            if transpose:
                volume = torch.transpose(volume, 3, 4)
            # aff: 3*z*y*x 
            vout = model(volume).detach().cpu()

            if transpose: # swap x-/y-affinity
                vout = torch.transpose(vout, 3, 4)
            if zflip:
                vout = torch.flip(vout, [2])
            if yflip:
                vout = torch.flip(vout, [3])
            if xflip:
                vout = torch.flip(vout, [4])
                
            # cast to numpy array
            vout = vout.numpy()
            if out is None:
                if self.mode == 'min':
                    out = np.ones(vout.shape, dtype=np.float32)
                elif self.mode == 'max':
                    out = np.zeros(vout.shape, dtype=np.float32)
                elif self.mode == 'mean':
                    out = np.zeros(vout.shape, dtype=np.float32)

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

    def update_name(self, name):
        extension = "_"
        if self.num_aug == 4:
            extension += "tz"
        else:
            extension += "tzyx"
        # Update the suffix of the output filename to indicate
        # the use of test-time data augmentation.
        name_list = name.split('.')
        new_filename = name_list[0] + extension + "." + name_list[1]
        return new_filename