from __future__ import print_function, division
import numpy as np
import random
import torch
import os

# TODO: combine with data/misc/*.py

####################################################################
## Process image stacks.
####################################################################

def count_volume(data_sz, vol_sz, stride):
    return 1 + np.ceil((data_sz - vol_sz) / stride.astype(float)).astype(int)

def crop_volume(data, sz, st=(0, 0, 0)):  # C*D*W*H, C=1
    st = np.array(st).astype(np.int32)
    return data[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

def crop_volume_mul(data, sz, st=(0, 0, 0)):  # C*D*W*H, for multi-channel input
    return data[:, st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]]

####################################################################
## Rebalancing.
####################################################################

def rebalance_binary_class(label, mask=None, alpha=1.0):
    """Binary-class rebalancing."""
    weight_factor = label.float().sum() / torch.prod(torch.tensor(label.size()).float())
    weight_factor = torch.clamp(weight_factor, min=1e-2)
    weight = alpha * label*(1-weight_factor)/weight_factor + (1-label)
    
    return weight_factor, weight

####################################################################
## Affinitize.
####################################################################

def check_volume(data):
    """Ensure that data is a numpy 3D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,...]
    elif data.ndim == 3:
        pass
    elif data.ndim == 4:
        assert data.shape[0]==1
        data = np.reshape(data, data.shape[-3:])
    else:
        raise RuntimeError('data must be a numpy 3D array')

    assert data.ndim==3
    return data

# def affinitize(img, dst=(1,1,1), dtype=np.float32):
#     """
#     Transform segmentation to an affinity map.

#     Args:
#         img: 3D indexed image, with each index corresponding to each segment.

#     Returns:
#         ret: an affinity map (4D tensor).
#     """
#     img = check_volume(img)
#     if ret is None:
#         ret = np.zeros(img.shape, dtype=dtype)

#     # Sanity check.
#     (dz,dy,dx) = dst
#     assert abs(dx) < img.shape[-1]
#     assert abs(dy) < img.shape[-2]
#     assert abs(dz) < img.shape[-3]
####################################################################
## tile to volume
####################################################################
def vast2Seg(seg):
    # convert to 24 bits
    return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)

def tileToVolume(tiles, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8, tile_st=[0,0], tile_ratio=1, resize_order=1, ndim=1, black=128):
    # x: column
    # y: row
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = tiles[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if '{' in pattern:
                    path = pattern.format(row=row+tile_st[0], column=column+tile_st[1])
                else:
                    path = pattern
                if not os.path.exists(path): 
                    #return None
                    patch = black*np.ones((tile_sz,tile_sz),dtype=dt)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        from imageio import imread
                        patch = imread(path)
                    if tile_ratio != 1:
                        # scipy.misc.imresize: only do uint8
                        from scipy.ndimage import zoom
                        patch = zoom(patch, [tile_ratio,tile_ratio,1], order=resize_order)
                    if patch.ndim==2:
                        patch=patch[:,:,None]
                
                # last tile may not be full
                xp0 = column * tile_sz
                xp1 = xp0 + patch.shape[1]
                #xp1 = (column+1) * tile_sz
                yp0 = row * tile_sz
                yp1 = yp0 + patch.shape[0]
                #yp1 = (row + 1) * tile_sz
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    sz = result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0].shape
                    if resize_order==0: # label
                        if ndim==1: # 1-channel coding
                            result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0,0].reshape(sz)
                        else: # 3-channel coding
                            result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = vast2Seg(patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]).reshape(sz)
                    else: # image
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0].reshape(sz)
    return result
