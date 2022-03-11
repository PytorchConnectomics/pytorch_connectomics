import numpy as np

####################################################################
## Process image stacks.
####################################################################

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):
    # must be (z, y, x) or (c, z, y, x) format 
    assert data.ndim in [3, 4]
    st = np.array(st).astype(np.int32)

    if data.ndim == 3:
        return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]
    else: # crop spatial dimensions
        return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]
