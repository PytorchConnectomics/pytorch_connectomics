from __future__ import print_function, division
from typing import Optional, List

import os
import h5py
import glob
import numpy as np
import imageio
from scipy.ndimage import zoom

def readh5(filename, dataset = None):
    fid = h5py.File(filename,'r')
    if dataset is None:
        dataset = list(fid)[0]
    return np.array(fid[dataset])

def readimg_as_vol(filename):
    img_suf = filename[filename.rfind('.')+1:]
    assert img_suf in ['png', 'tif']
    data = imageio.imread(filename)
    return data[np.newaxis, :, :]

def readvol(filename, dataset=None):
    r"""Load a image volume in HDF5, TIFF or PNG formats.
    """
    img_suf = filename[filename.rfind('.')+1:]
    if img_suf in ['h5', 'hdf5']:
        data = readh5(filename, dataset)
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
    elif 'png' in img_suf:
        data = readimgs(filename)
    else:
        raise ValueError('unrecognizable file format for %s'%(filename))
    return data

def savevol(filename, vol, dataset='main', format='h5'):
    if format == 'h5':
        writeh5(filename, vol, dataset='main')         
    if format == 'png':
        currentDirectory = os.getcwd()
        img_save_path = os.path.join(currentDirectory, filename)
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        for i in range(vol.shape[0]):
            imageio.imsave('%s/%04d.png' % (img_save_path, i), vol[i])

def readim(filename, do_channel=False):
    # x,y,c
    if not os.path.exists(filename): 
        im = None
    else:# note: cv2 do "bgr" channel order
        im = imageio.imread(filename)
        if do_channel and im.ndim==2:
            im=im[:,:,None]
    return im

def readimgs(filename):
    filelist = sorted(glob.glob(filename))
    num_imgs = len(filelist)

    # decide numpy array shape:
    img = imageio.imread(filelist[0])
    data = np.zeros((num_imgs, img.shape[0], img.shape[1]), dtype=np.uint8)
    data[0] = img

    # load all images
    if num_imgs > 1:
        for i in range(1, num_imgs):
            data[i] = imageio.imread(filelist[i])

    return data

def writeh5(filename, dtarray, dataset='main'):
    fid=h5py.File(filename,'w')
    if isinstance(dataset, (list,)):
        for i,dd in enumerate(dataset):
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]
    else:
        ds = fid.create_dataset(dataset, dtarray.shape, compression="gzip", dtype=dtarray.dtype)
        ds[:] = dtarray
    fid.close()

                                                                               
####################################################################
## tile to volume
####################################################################

def vast2Seg(seg):
    # convert to 24 bits
    if seg.ndim==2 or seg.shape[-1]==1:
        return np.squeeze(seg)
    elif seg.ndim == 3: # 1 rgb image
        return seg[:,:,0].astype(np.uint32)*65536+seg[:,:,1].astype(np.uint32)*256+seg[:,:,2].astype(np.uint32)
    elif seg.ndim == 4: # n rgb image
        return seg[:,:,:,0].astype(np.uint32)*65536+seg[:,:,:,1].astype(np.uint32)*256+seg[:,:,:,2].astype(np.uint32)

def tile2volume(tiles: List[str], coord: List[int], coord_m: List[int], tile_sz: int, 
                dt: type=np.uint8, tile_st: List[int]=[0,0], tile_ratio: float=1.0, 
                do_im: bool=True, background: int=128) -> np.ndarray:
    """Construct a volume from image tiles based on the given volume coordinate.

    Args:
        tiles (List[str]): a list of paths to the image tiles.
        coord (List[int]): the coordinate of the volume to be constructed.
        coord_m (List[int]): the coordinate of the whole dataset with the tiles.
        tile_sz (int): the height and width of the tiles, which is assumed to be square.
        dt (type): data type of the constructed volume. Default: numpy.uint8
        tile_st (List[int]): start position of the tiles. Default: [0, 0]
        tile_ratio (float): scale factor for resizing the tiles. Default: 1.0
        do_im (bool): construct an image volume (apply linear interpolation for resizing). Default: `True`
        background (int): background value for filling the constructed volume. Default: 128
    """
    z0o, z1o, y0o, y1o, x0o, x1o = coord # region to crop
    z0m, z1m, y0m, y1m, x0m, x1m = coord_m # tile boundary

    bd = [max(-z0o,z0m), max(0,z1o-z1m), max(-y0o,y0m), max(0,y1o-y1m), max(-x0o,x0m), max(0,x1o-x1m)]
    z0, y0, x0 = max(z0o,z0m), max(y0o,y0m), max(x0o,x0m)
    z1, y1, x1 = min(z1o,z1m), min(y1o,y1m), min(x1o,x1m)

    result = background*np.ones((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz # floor
    c1 = (x1 + tile_sz-1) // tile_sz # ceil
    r0 = y0 // tile_sz
    r1 = (y1 + tile_sz-1) // tile_sz
    for z in range(z0, z1):
        pattern = tiles[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                if r'{row}_{column}' in pattern:
                    path = pattern.format(row=row+tile_st[0], column=column+tile_st[1])
                else:
                    path = pattern
                patch = readim(path, do_channel=True)
                if patch is not None:
                    if tile_ratio != 1: # im ->1, label->0 
                        patch = zoom(patch, [tile_ratio,tile_ratio,1], order=int(do_im))
                
                    # last tile may not be of the same size
                    xp0 = column * tile_sz
                    xp1 = xp0 + patch.shape[1]
                    yp0 = row * tile_sz
                    yp1 = yp0 + patch.shape[0]
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    if do_im:# image
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0,0]
                    else: # label
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = vast2Seg(patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])

    # For chunks touching the border of the large input volume, apply padding.                    
    if max(bd)>0:
        result = np.pad(result,((bd[0], bd[1]),(bd[2], bd[3]),(bd[4], bd[5])), 'reflect')
    return result
