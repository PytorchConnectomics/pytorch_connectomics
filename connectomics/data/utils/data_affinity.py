import numpy as np

# from Janelia pyGreentea
# https://github.com/naibaf7/PyGreentea
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
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood 

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

def seg_to_aff(seg, nhood=mknhood3d(1), pad='replicate'):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.float32)
    
    if len(shape) == 3: # 3D affinity
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
    elif len(shape) == 2: # 2D affinity
        for e in range(nEdge):
            aff[e, \
                max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] = \
                            (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] == \
                             seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] ) \
                            * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                                max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1])] > 0 ) \
                            * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                                max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1])] > 0 )

    if nEdge==3 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)
        aff[2,:,:,0] = (seg[:,:,0]>0).astype(aff.dtype)
    elif nEdge==2 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)

    return aff
