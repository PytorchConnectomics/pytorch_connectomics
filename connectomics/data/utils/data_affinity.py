import numpy as np

__all__ = [
    "seg2aff_v0",
    "seg2aff_v1",
    "seg2aff_v2",
]


def mknhood2d(radius: int=1):
    # Makes nhood structures for some most used dense graphs.
    # Janelia pyGreentea: https://github.com/naibaf7/PyGreentea

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad+1, 1)
    y = np.arange(-ceilrad, ceilrad+1, 1)
    [i, j] = np.meshgrid(y, x)

    idxkeep = (i**2+j**2) <= radius**2
    i = i[idxkeep].ravel()
    j = j[idxkeep].ravel()
    zeroIdx = np.ceil(len(i)/2).astype(np.int32)

    nhood = np.vstack((i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood


def mknhood3d(radius: int=1):
    # Makes nhood structures for some most used dense graphs.
    # Janelia pyGreentea: https://github.com/naibaf7/PyGreentea

    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad, ceilrad+1, 1)
    y = np.arange(-ceilrad, ceilrad+1, 1)
    z = np.arange(-ceilrad, ceilrad+1, 1)
    [i, j, k] = np.meshgrid(z, y, x)

    idxkeep = (i**2+j**2+k**2) <= radius**2
    i = i[idxkeep].ravel()
    j = j[idxkeep].ravel()
    k = k[idxkeep].ravel()
    zeroIdx = np.array(len(i) // 2).astype(np.int32)

    nhood = np.vstack(
        (k[:zeroIdx], i[:zeroIdx], j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))


def mknhood3d_aniso(radiusxy: int=1, radiusxy_zminus1: float=1.8):
    # Makes nhood structures for some most used dense graphs.
    # Janelia pyGreentea: https://github.com/naibaf7/PyGreentea

    nhoodxyz = mknhood3d(radiusxy)
    nhoodxy_zminus1 = mknhood2d(radiusxy_zminus1)
    nhood = np.zeros(
        (nhoodxyz.shape[0]+2*nhoodxy_zminus1.shape[0], 3), dtype=np.int32)
    nhood[:3, :3] = nhoodxyz
    nhood[3:, 0] = -1
    nhood[3:, 1:] = np.vstack((nhoodxy_zminus1, -nhoodxy_zminus1))

    return np.ascontiguousarray(nhood)


def seg_to_aff(seg, nhood=mknhood3d(1), pad='replicate'):
    # Constructs an affinity graph from a segmentation
    # Janelia pyGreentea: https://github.com/naibaf7/PyGreentea

    # Assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape, dtype=np.float32)

    if len(shape) == 3:  # 3D affinity
        for e in range(nEdge):
            aff[e,
                max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1]),
                max(0, -nhood[e, 2]):min(shape[2], shape[2]-nhood[e, 2])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1]),
                     max(0, -nhood[e, 2]):min(shape[2], shape[2]-nhood[e, 2])] ==
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0]+nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1]+nhood[e, 1]),
                     max(0, nhood[e, 2]):min(shape[2], shape[2]+nhood[e, 2])]) \
                * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                       max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1]),
                       max(0, -nhood[e, 2]):min(shape[2], shape[2]-nhood[e, 2])] > 0) \
                * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0]+nhood[e, 0]),
                       max(0, nhood[e, 1]):min(shape[1], shape[1]+nhood[e, 1]),
                       max(0, nhood[e, 2]):min(shape[2], shape[2]+nhood[e, 2])] > 0)
    elif len(shape) == 2:  # 2D affinity
        for e in range(nEdge):
            aff[e,
                max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1])] = \
                (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                     max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1])] ==
                 seg[max(0, nhood[e, 0]):min(shape[0], shape[0]+nhood[e, 0]),
                     max(0, nhood[e, 1]):min(shape[1], shape[1]+nhood[e, 1])]) \
                * (seg[max(0, -nhood[e, 0]):min(shape[0], shape[0]-nhood[e, 0]),
                       max(0, -nhood[e, 1]):min(shape[1], shape[1]-nhood[e, 1])] > 0) \
                * (seg[max(0, nhood[e, 0]):min(shape[0], shape[0]+nhood[e, 0]),
                       max(0, nhood[e, 1]):min(shape[1], shape[1]+nhood[e, 1])] > 0)

    if nEdge == 3 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)
        aff[2, :, :, 0] = (seg[:, :, 0] > 0).astype(aff.dtype)
    elif nEdge == 2 and pad == 'replicate':  # pad the boundary affinity
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)

    return aff


def seg2aff_v0(seg, pad='replicate'):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x) or (e, y, x)

    assert seg.ndim in [2, 3]
    mknhood_func = mknhood3d if seg.ndim == 3 else mknhood2d
    nhood = mknhood_func(1)

    shape = seg.shape
    nEdge = nhood.shape[0]
    # shape = (aff channel, z, y, x)
    aff = np.zeros((nEdge,)+shape, dtype=np.float32)

    for e in range(nEdge):
        offset_0 = nhood[e, 0]
        offset_1 = nhood[e, 1]

        # position on seg array
        start_0 = max(0, -offset_0)
        start_1 = max(0, -offset_1)
        end_0 = min(shape[0], shape[0]-offset_0)
        end_1 = min(shape[1], shape[1]-offset_1)

        # position on offset array
        offstart_0 = max(0, offset_0)
        offstart_1 = max(0, offset_1)
        offend_0 = min(shape[0], shape[0]+offset_0)
        offend_1 = min(shape[1], shape[1]+offset_1)

        if len(shape) == 2:
            seg_array = seg[start_0:end_0, start_1:end_1]
            offset_array = seg[offstart_0:offend_0, offstart_1:offend_1]
            # assign value to aff
            aff[e, start_0:end_0, start_1:end_1] = (
                seg_array == offset_array) * (seg_array > 0) * (offset_array > 0)

        elif len(shape) == 3:
            # additional dimension
            offset_2 = nhood[e, 2]
            start_2 = max(0, -offset_2)
            end_2 = min(shape[2], shape[2]-offset_2)
            offstart_2 = max(0, offset_2)
            offend_2 = min(shape[2], shape[2]+offset_2)

            seg_array = seg[start_0:end_0, start_1:end_1, start_2:end_2]
            offset_array = seg[offstart_0:offend_0,
                               offstart_1:offend_1, offstart_2:offend_2]
            # assgin value to aff
            # print(seg_array.shape, offset_array.shape)
            aff[e, start_0:end_0, start_1:end_1, start_2:end_2] = (
                seg_array == offset_array) * (seg_array > 0) * (offset_array > 0)

    # pad the boundary affinity, assuming offset is [[-1,  0,  0],[ 0, -1,  0],[ 0,  0, -1]]
    if nEdge == 3 and pad == 'replicate':
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)
        aff[2, :, :, 0] = (seg[:, :, 0] > 0).astype(aff.dtype)
    # pad the boundary affinity, assuming offset is [[-1,  0],[ 0, -1]]
    elif nEdge == 2 and pad == 'replicate':
        aff[0, 0] = (seg[0] > 0).astype(aff.dtype)
        aff[1, :, 0] = (seg[:, 0] > 0).astype(aff.dtype)

    return aff


def seg2aff_v1(seg: np.ndarray,
               dz: int = 1,
               dy: int = 1,
               dx: int = 1,
               padding: str = 'edge') -> np.array:
    # Calaulate long range affinity. Output: (affs, z, y, x)

    shape = seg.shape
    z_1 = slice(dz, None)
    y_1 = slice(dy, None)
    x_1 = slice(dx, None)

    z_2 = slice(None, dz)
    y_2 = slice(None, dy)
    x_2 = slice(None, dx)

    z_3 = slice(None, -dz)
    y_3 = slice(None, -dy)
    x_3 = slice(None, -dx)

    if seg.ndim == 3:
        aff = np.zeros((3,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dz, 0), (dy, 0), (dx, 0)), 'edge')
            # print(seg_pad.shape)
            aff[2] = (seg == seg_pad[z_1, y_1, x_3]) * \
                (seg != 0) * (seg_pad[z_1, y_1, x_3] != 0)
            aff[1] = (seg == seg_pad[z_1, y_3, x_1]) * \
                (seg != 0) * (seg_pad[z_1, y_3, x_1] != 0)
            aff[0] = (seg == seg_pad[z_3, y_1, x_1]) * \
                (seg != 0) * (seg_pad[z_3, y_1, x_1] != 0)

        else:
            aff[2, :, :, x_1] = (seg[:, :, x_1] == seg[:, :, x_3]) * \
                (seg[:, :, x_1] != 0) * (seg[:, :, x_3] != 0)
            aff[1, :, y_1, :] = (seg[:, y_1, :] == seg[:, y_3, :]) * \
                (seg[:, y_1, :] != 0) * (seg[:, y_3, :] != 0)
            aff[0, z_1, :, :] = (seg[z_1, :, :] == seg[z_3, :, :]) * \
                (seg[z_1, :, :] != 0) * (seg[z_3, :, :] != 0)
            if padding == 'replicate':
                aff[2, :, :, x_2] = (seg[:, :, x_2] != 0).astype(aff.dtype)
                aff[1, :, y_2, :] = (seg[:, y_2, :] != 0).astype(aff.dtype)
                aff[0, z_2, :, :] = (seg[z_2, :, :] != 0).astype(aff.dtype)

    elif seg.ndim == 2:
        aff = np.zeros((2,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dy, 0), (dx, 0)), 'edge')
            # print(seg_pad.shape)
            aff[1] = (seg == seg_pad[y_1, x_3]) * \
                (seg != 0) * (seg_pad[y_1, x_3] != 0)
            aff[0] = (seg == seg_pad[y_3, x_1]) * \
                (seg != 0) * (seg_pad[y_3, x_1] != 0)

        else:
            aff[1, :, x_1] = (seg[:, x_1] == seg[:, x_3]) * \
                (seg[:, x_1] != 0) * (seg[:, x_3] != 0)
            aff[0, y_1, :] = (seg[y_1, :] == seg[y_3, :]) * \
                (seg[y_1, :] != 0) * (seg[y_3, :] != 0)
            if padding == 'replicate':
                aff[1, :, x_2] = (seg[:, x_2] != 0).astype(aff.dtype)
                aff[0, y_2, :] = (seg[y_2, :] != 0).astype(aff.dtype)

    return aff


def seg2aff_v2(seg: np.ndarray,
               dz: int = 1,
               dy: int = 1,
               dx: int = 1,
               padding: str = 'edge') -> np.array:
    # Calaulate long range affinity. Output: (affs, z, y, x)

    shape = seg.shape
    z_1 = slice(dz, -dz)
    y_1 = slice(dy, -dy)
    x_1 = slice(dx, -dx)

    z_2 = slice(None, dz)
    y_2 = slice(None, dy)
    x_2 = slice(None, dx)

    z_3 = slice(None, -2*dz)
    y_3 = slice(None, -2*dy)
    x_3 = slice(None, -2*dx)

    z_4 = slice(2*dz, None)
    y_4 = slice(2*dy, None)
    x_4 = slice(2*dx, None)

    z_5 = slice(-dz, None)
    y_5 = slice(-dy, None)
    x_5 = slice(-dx, None)

    if seg.ndim == 3:
        aff = np.zeros((3,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dz, dz), (dy, dy), (dx, dx)), 'edge')
            # print(seg_pad.shape)
            aff[2] = (seg_pad[z_1, y_1, x_3] == seg_pad[z_1, y_1, x_4]) * \
                (seg_pad[z_1, y_1, x_3] != 0) * (seg_pad[z_1, y_1, x_4] != 0)
            aff[1] = (seg_pad[z_1, y_3, x_1] == seg_pad[z_1, y_4, x_1]) * \
                (seg_pad[z_1, y_3, x_1] != 0) * (seg_pad[z_1, y_4, x_1] != 0)
            aff[0] = (seg_pad[z_3, y_1, x_1] == seg_pad[z_4, y_1, x_1]) * \
                (seg_pad[z_3, y_1, x_1] != 0) * (seg_pad[z_4, y_1, x_1] != 0)

        else:
            aff[2, :, :, x_1] = (seg[:, :, x_3] == seg[:, :, x_4]) * \
                (seg[:, :, x_3] != 0) * (seg[:, :, x_4] != 0)
            aff[1, :, y_1, :] = (seg[:, y_3, :] == seg[:, y_4, :]) * \
                (seg[:, y_3, :] != 0) * (seg[:, y_4, :] != 0)
            aff[0, z_1, :, :] = (seg[z_3, :, :] == seg[z_4, :, :]) * \
                (seg[z_3, :, :] != 0) * (seg[z_4, :, :] != 0)
            if padding == 'replicate':
                aff[2, :, :, x_2] = (seg[:, :, x_2] != 0).astype(aff.dtype)
                aff[1, :, y_2, :] = (seg[:, y_2, :] != 0).astype(aff.dtype)
                aff[0, z_2, :, :] = (seg[z_2, :, :] != 0).astype(aff.dtype)
                aff[2, :, :, x_5] = (seg[:, :, x_5] != 0).astype(aff.dtype)
                aff[1, :, y_5, :] = (seg[:, y_5, :] != 0).astype(aff.dtype)
                aff[0, z_5, :, :] = (seg[z_5, :, :] != 0).astype(aff.dtype)

    elif seg.ndim == 2:
        aff = np.zeros((2,) + shape, dtype=np.float32)
        if padding == 'edge':
            seg_pad = np.pad(seg, ((dy, dy), (dx, dx)), 'edge')
            # print(seg_pad.shape)
            aff[1] = (seg_pad[y_1, x_3] == seg_pad[y_1, x_4]) * \
                (seg_pad[y_1, x_3] != 0) * (seg_pad[y_1, x_4] != 0)
            aff[0] = (seg_pad[y_3, x_1] == seg_pad[y_4, x_1]) * \
                (seg_pad[y_3, x_1] != 0) * (seg_pad[y_4, x_1] != 0)

        else:
            aff[1, :, x_1] = (seg[:, x_3] == seg[:, x_4]) * \
                (seg[:, x_3] != 0) * (seg[:, x_4] != 0)
            aff[0, y_1, :] = (seg[y_3, :] == seg[y_4, :]) * \
                (seg[y_3, :] != 0) * (seg[y_4, :] != 0)
            if padding == 'replicate':
                aff[1, :, x_2] = (seg[:, x_2] != 0).astype(aff.dtype)
                aff[0, y_2, :] = (seg[y_2, :] != 0).astype(aff.dtype)
                aff[1, :, x_5] = (seg[:, x_5] != 0).astype(aff.dtype)
                aff[0, y_5, :] = (seg[y_5, :] != 0).astype(aff.dtype)

    return aff
