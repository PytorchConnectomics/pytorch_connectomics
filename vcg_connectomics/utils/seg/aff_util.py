import numpy as np
from scipy.misc import comb
import scipy.sparse


def affinitize(img, ret=None, dst=(1,1,1), dtype='float32'):
    # PNI code
    """
    Transform segmentation to an affinity map.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: an affinity map (4D tensor).
    """
    img = check_volume(img)
    if ret is None:
        ret = np.zeros(img.shape, dtype=dtype)

    # Sanity check.
    (dz,dy,dx) = dst
    assert abs(dx) < img.shape[-1]
    assert abs(dy) < img.shape[-2]
    assert abs(dz) < img.shape[-3]

    # Slices.
    s0 = list()
    s1 = list()
    s2 = list()
    for i in range(3):
        if dst[i] == 0:
            s0.append(slice(None))
            s1.append(slice(None))
            s2.append(slice(None))
        elif dst[i] > 0:
            s0.append(slice(dst[i],  None))
            s1.append(slice(dst[i],  None))
            s2.append(slice(None, -dst[i]))
        else:
            s0.append(slice(None,  dst[i]))
            s1.append(slice(-dst[i], None))
            s2.append(slice(None,  dst[i]))

    ret[s0] = (img[s1]==img[s2]) & (img[s1]>0)
    return ret[np.newaxis,...]

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


