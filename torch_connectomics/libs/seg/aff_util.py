import numpy as np
import scipy.sparse
try:  # SciPy >= 0.19
    from scipy.special import comb, logsumexp
except ImportError:
    from scipy.misc import comb, logsumexp  # noqa

def affinitize(img, dst=(1,1,1), dtype=np.float32):
    """
    Transform segmentation to an affinity map.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        aff: an affinity map with shape in (c, z, y, x).
    """
    (dz,dy,dx) = dst
    assert abs(dx) < img.shape[-1]
    assert abs(dy) < img.shape[-2]
    assert abs(dz) < img.shape[-3]

    z_pad = np.pad(img, ((0, dz), (0, 0), (0, 0)), mode='edge')
    z_aff = np.diff(z_pad, n=dz, axis=0)
    z_aff = (z_aff==0).astype(int)

    y_pad = np.pad(img, ((0, 0), (0, dy), (0, 0)), mode='edge')
    y_aff = np.diff(y_pad, n=dy, axis=1)
    y_aff = (y_aff==0).astype(int)

    x_pad = np.pad(img, ((0, 0), (0, 0), (0, dx)), mode='edge')
    x_aff = np.diff(x_pad, n=dx, axis=2)
    x_aff = (x_aff==0).astype(int)

    assert z_aff.shape == img.shape
    assert y_aff.shape == img.shape
    assert x_aff.shape == img.shape

    z_aff[np.where(img==0)] = 0
    y_aff[np.where(img==0)] = 0
    x_aff[np.where(img==0)] = 0

    return np.stack([z_aff, y_aff, x_aff], 0).astype(dtype)

def bmap_to_affgraph(bmap,nhood,return_min_idx=False):
    # constructs an affinity graph from a boundary map
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = bmap.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)
    minidx = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = np.minimum( \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])], \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] )
        minidx[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]
    return aff

def seg_to_affgraph(seg,nhood,pad=''):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )
    if nEdge==3 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)                                                                  
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)                                                              
        aff[2,:,:,0] = (seg[:,:,0]>0).astype(aff.dtype)   

    return aff

def affgraph_to_edgelist(aff,nhood):
    node1,node2 = nodelist_like(aff.shape[1:],nhood)
    return (node1.ravel(),node2.ravel(),aff.ravel())

def nodelist_like(shape,nhood):
    # constructs the node lists corresponding to the edge list representation of an affinity graph
    # assume  node shape is represented as:
    # shape = (z, y, x)
    # nhood.shape = (edges, 3)
    nEdge = nhood.shape[0]
    nodes = np.arange(np.prod(shape),dtype=np.uint64).reshape(shape)
    node1 = np.tile(nodes,(nEdge,1,1,1))
    node2 = np.full(node1.shape,-1,dtype=np.uint64)

    for e in range(nEdge):
        node2[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                nodes[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                     max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                     max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])]

    return (node1, node2)


