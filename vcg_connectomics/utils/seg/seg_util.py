import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.morphology import binary_erosion,binary_dilation

# reduce the labeling
def relabel(segmentation):
    # get the unique labels
    uid = np.unique(segmentation)
    # get the maximum label for the segment
    mid = int(uid.max()) + 1

    # create an array from original segment id to reduced id
    mapping = np.zeros(mid, dtype=segmentation.dtype)
    mapping[uid] = np.arange(len(uid), dtype=segmentation.dtype)
    return mapping[segmentation]

def remove_small(seg, thres=100):                                                                    
    sz = seg.shape                                                                                   
    seg = seg.reshape(-1)                                                                            
    uid, uc = np.unique(seg, return_counts=True)                                                     
    seg[np.in1d(seg,uid[uc<thres])] = 0                                                              
    return seg.reshape(sz)

def countVolume(data_sz, vol_sz, stride):
    return 1+np.ceil((data_sz - vol_sz) / stride.astype(np.float32)).astype(int)

def mk_cont_table(seg1,seg2):
    cont_table = coo_matrix((np.ones(seg1.shape),(seg1,seg2))).toarray()
    return cont_table

def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)

    idxkeep = (i**2+j**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    z = np.arange(-ceilrad,ceilrad+1,1)
    [i,j,k] = np.meshgrid(z,y,x)

    idxkeep = (i**2+j**2+k**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); k=k[idxkeep].ravel();
    zeroIdx = np.array(len(i) // 2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))

def mknhood3d_aniso(radiusxy=1,radiusxy_zminus1=1.8):
    # Makes nhood structures for some most used dense graphs.
    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
    nhood = np.zeros((nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0],3),dtype=np.int32)
    nhood[:3,:3] = nhoodxyz
    nhood[3:,0] = -1
    nhood[3:,1:] = np.vstack((nhoodxy_zminus1,-nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)

# https://github.com/torms3/DataProvider/blob/refactoring/python/dataprovider/utils.py
### check input shape
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


def check_tensor(data):
    """Ensure that data is a numpy 4D array."""
    assert isinstance(data, np.ndarray)

    if data.ndim == 2:
        data = data[np.newaxis,np.newaxis,...]
    elif data.ndim == 3:
        data = data[np.newaxis,...]
    elif data.ndim == 4:
        pass
    else:
        raise RuntimeError('data must be a numpy 4D array')

    assert data.ndim==4
    return data


def fill_data(shape, filler={'type':'zero'}, dtype='float32'):
    """
    Return numpy array of shape, filled with specified values.
    Args:
        shape: Array shape.
        filler: {'type':'zero'} (default)
                {'type':'one'}
                {'type':'constant', 'value':%f}
                {'type':'gaussian', 'loc':%f, 'scale':%f}
                {'type':'uniform', 'low':%f, 'high':%f}
                {'type':'randi', 'low':%d, 'high':%d}
    Returs:
        data: Numpy array of shape, filled with specified values.
    """
    data = np.zeros(shape, dtype=dtype)

    assert 'type' in filler
    if filler['type'] == 'zero':
        # Fill zeros.
        pass
    elif filler['type'] == 'one':
        # Fill ones.
        data = np.ones(shape, dtype=dtype)
    elif filler['type'] == 'constant':
        # Fill constant value.
        assert 'value' in filler
        data[:] = filler['value']
    elif filler['type'] == 'gaussian':
        # Fill random numbers from Gaussian(loc, scale).
        loc = filler.get('mean', 0.0)
        scale = filler.get('std', 1.0)
        data[:] = np.random.normal(loc=loc, scale=scale, size=shape)
    elif filler['type'] == 'uniform':
        # Fill random numbers from Uniform(low, high).
        low = filler.get('low', 0.0)
        high = filler.get('high', 1.0)
        data[:] = np.random.uniform(low=low, high=high, size=shape)
    elif filler['type'] == 'randint':
        low = filler.get('low', 0)
        high = filler.get('high', None)
        data[:] = np.random.randint(low=low, high=high, size=shape)
    else:
        raise RuntimeError('invalid filler type [%s]' % filler['type'])
    return data

def genSegMalis(gg3,iter_num): # given input seg map, widen the seg border
    gg3_dz = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dz[1:,:,:] = (np.diff(gg3,axis=0))
    gg3_dy = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dy[:,1:,:] = (np.diff(gg3,axis=1))
    gg3_dx = np.zeros(gg3.shape).astype(np.uint32)
    gg3_dx[:,:,1:] = (np.diff(gg3,axis=2))
    gg3g = ((gg3_dx+gg3_dy)>0)
    #stel=np.array([[1, 1],[1,1]]).astype(bool)
    stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
    #stel=np.array([[1,1,1,1],[1, 1, 1, 1],[1,1,1,1],[1,1,1,1]]).astype(bool)
    gg3gd=np.zeros(gg3g.shape)
    for i in range(gg3g.shape[0]):
        gg3gd[i,:,:]=binary_dilation(gg3g[i,:,:],structure=stel,iterations=iter_num)
    out = gg3.copy()
    out[gg3gd==1]=0
    return out

def markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid 
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel=np.array([[1,1,1], [1,1,1]]).astype(bool)
        if len(seg.shape)==2:
            out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
            seg[out==0] = -1
        else: # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z] # by reference
                out = binary_dilation(tmp>0, structure=stel, iterations=iter_num)
                tmp[out==0] = -1
    else:
        stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
        out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
        seg[out==0] = -1
    return seg
