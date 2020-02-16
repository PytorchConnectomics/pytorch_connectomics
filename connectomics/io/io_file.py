import h5py
import numpy as np

def readh5(filename, dataset=''):
    fid = h5py.File(filename,'r')
    if dataset=='':
        dataset = list(fid)[0]
    return np.array(fid[dataset])


def readvol(filename, dataset=''):
    img_suf = filename[filename.rfind('.')+1:]
    if img_suf == 'h5':
        data = readh5(filename, dataset)
    elif 'tif' in img_suf:
        data = imageio.volread(filename).squeeze()
    else:
        raise ValueError('unrecognizable file format for %s'%(filename))

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
