from __future__ import print_function, division
from typing import Optional, Union, List

import torch
import scipy
import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import erosion, dilation, disk
from skimage.measure import label as label_cc  # avoid namespace conflict
from scipy.signal import convolve2d

from .data_affinity import *
from .data_transform import *

RATES_TYPE = Optional[Union[List[int], int]]


def getSegType(mid):
    # reduce the label dtype
    m_type = np.uint64
    if mid < 2**8:
        m_type = np.uint8
    elif mid < 2**16:
        m_type = np.uint16
    elif mid < 2**32:
        m_type = np.uint32
    return m_type


def relabel(seg, do_type=False):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-background samples
    if len(uid) == 1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1  # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    if do_type:
        m_type = getSegType(mid)
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]


def remove_small(seg, thres=100):
    sz = seg.shape
    seg = seg.reshape(-1)
    uid, uc = np.unique(seg, return_counts=True)
    seg[np.in1d(seg, uid[uc < thres])] = 0
    return seg.reshape(sz)


def im2col(A, BSZ, stepsize=1):
    # Parameters
    M, N = A.shape
    # Get Starting block indices
    start_idx = np.arange(
        0, M-BSZ[0]+1, stepsize)[:, None]*N + np.arange(0, N-BSZ[1]+1, stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:, None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())


def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4):
    # "we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing
    # more than one positive segment ID (zero is reserved for background) is marked as background."
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz) == 3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(
                np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
            p0 = patch.max(axis=1)
            patch[patch == 0] = mm+1
            p1 = patch.min(axis=1)
            seg[z] = seg[z]*((p0 == p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(
            np.pad(seg, ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis=1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg


def seg_to_small_seg(seg, thres=25, rr=2):
    # rr: z/x-y resolution ratio
    sz = seg.shape
    mask = np.zeros(sz, np.uint8)
    for z in np.where(seg.max(axis=1).max(axis=1) > 0)[0]:
        tmp = label_cc(seg[z])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1]+1, np.uint8)
        rl[ui[uc < thres]] = 1
        rl[0] = 0
        mask[z] += rl[tmp]
    for y in np.where(seg.max(axis=2).max(axis=0) > 0)[0]:
        tmp = label_cc(seg[:, y])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1]+1, np.uint8)
        rl[ui[uc < thres//rr]] = 1
        rl[0] = 0
        mask[:, y] += rl[tmp]
    for x in np.where(seg.max(axis=0).max(axis=0) > 0)[0]:
        tmp = label_cc(seg[:, :, x])
        ui, uc = np.unique(tmp, return_counts=True)
        rl = np.zeros(ui[-1]+1, np.uint8)
        rl[ui[uc < thres//rr]] = 1
        rl[0] = 0
        mask[:, :, x] += rl[tmp]
    return mask


def seg_to_instance_bd(seg: np.ndarray,
                       tsz_h: int = 1,
                       do_bg: bool = True,
                       do_convolve: bool = True) -> np.ndarray:
    """Generate instance contour map from segmentation masks.

    Args:
        seg (np.ndarray): segmentation map (3D array is required).
        tsz_h (int, optional): size of the dilation struct. Defaults: 1
        do_bg (bool, optional): generate contour between instances and background. Defaults: True
        do_convolve (bool, optional): convolve with edge filters. Defaults: True

    Returns:
        np.ndarray: binary instance contour map.

    Note:
        According to the experiment on the Lucchi mitochondria segmentation dastaset, convolving
        the edge filters with segmentation masks to generate the contour map is about 3x larger
        then using the `im2col` function. However, calculating the contour between only non-background
        instances is not supported under the convolution mode.
    """
    if do_bg == False:
        do_convolve = False
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    tsz = tsz_h*2+1

    if do_convolve:
        sobel = [1, 0, -1]
        sobel_x = np.array(sobel).reshape(3, 1)
        sobel_y = np.array(sobel).reshape(1, 3)
        for z in range(sz[0]):
            slide = seg[z]
            edge_x = convolve2d(slide, sobel_x, 'same', boundary='symm')
            edge_y = convolve2d(slide, sobel_y, 'same', boundary='symm')
            edge = np.maximum(np.abs(edge_x), np.abs(edge_y))
            contour = (edge != 0).astype(np.uint8)
            bd[z] = dilation(contour, np.ones((tsz, tsz), dtype=np.uint8))
        return bd

    mm = seg.max()
    for z in range(sz[0]):
        patch = im2col(
            np.pad(seg[z], ((tsz_h, tsz_h), (tsz_h, tsz_h)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        if do_bg:  # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0 > 0)*(p0 != p1)).reshape(sz[1:])
        else:  # between two non-zero seg
            patch[patch == 0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0 != 0)*(p1 != 0)*(p0 != p1)).reshape(sz[1:])
    return bd


def markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel = np.array([[1, 1, 1], [1, 1, 1]]).astype(bool)
        if len(seg.shape) == 2:
            out = binary_dilation(seg > 0, structure=stel, iterations=iter_num)
            seg[out == 0] = -1
        else:  # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z]  # by reference
                out = binary_dilation(
                    tmp > 0, structure=stel, iterations=iter_num)
                tmp[out == 0] = -1
    else:
        stel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(bool)
        out = binary_dilation(seg > 0, structure=stel, iterations=iter_num)
        seg[out == 0] = -1
    return seg


def seg2binary(label, topt):
    if len(topt) == 1:
        return label > 0

    fg_mask = np.zeros_like(label).astype(bool)
    _, *fg_indices = topt.split('-')
    for fg in fg_indices:
        fg_mask = np.logical_or(fg_mask, label == int(fg))

    return fg_mask


def seg2affinity(label, topt):
    assert label.ndim in [2, 3], \
        'Undefined affinity for ndim=' + str(label.ndim)
    if len(topt) == 1:
        return seg2aff_v0(label)

    aff_func_dict = {
        'v1': seg2aff_v1,
        'v2': seg2aff_v2,
    }

    # valid format: 2-z-y-x-version
    options = topt.split('-')
    assert len(options) == 5
    _, z, y, x, version = options
    return aff_func_dict[version](
        label, int(z), int(y), int(x))


def erode_label(label: np.ndarray,
                index: int,
                erosion_rates: RATES_TYPE = None):

    if erosion_rates is None:
        return label

    label_erosion = erosion_rates
    if isinstance(label_erosion, list):
        label_erosion = label_erosion[index]
    return seg_widen_border(label, label_erosion)


def dilate_label(label: np.ndarray,
                 index: int,
                 dilation_rates: RATES_TYPE = None):
    if dilation_rates is None:
        return label

    label_dilation = dilation_rates
    if isinstance(label_dilation, list):
        label_dilation = label_dilation[index]

    tsz = 2*label_dilation + 1
    assert label.ndim in [2, 3]
    shape = (1, tsz, tsz) if label.ndim == 3 else (tsz, tsz)
    return dilation(label, np.ones(shape, dtype=label.dtype))


def seg2polarity(label):
    # segmentation to 3-channel synaptic polarity masks
    tmp = [None]*3
    tmp[0] = np.logical_and((label % 2) == 1, label > 0)
    tmp[1] = np.logical_and((label % 2) == 0, label > 0)
    tmp[2] = (label > 0)
    return np.stack(tmp, 0).astype(np.float32)

def seg2inst_edt(label, topt):
    # Format of the target option: 5-a-b-c-d
    # a: mode, b: padding, c: quantize, d: z_resolution
    if len(topt) == 1:
        topt = topt + '-2d-0-0-5.0' # 2d w/o padding or quantize (default)

    _, mode, padding, quant, z_res = topt.split('-')
    resolution = (float(z_res), 1.0, 1.0)
    return edt_instance(label.copy(), mode, resolution=resolution,
                        quantize=bool(int(quant)), padding=bool(int(padding)))



def extend_centers(neighbors, centers, isneighbor, Ly, Lx, n_iter=200, device=torch.device('cuda')):
    # Adapted from https://github.com/MouseLand/cellpose
    """ runs diffusion to generate flows for training images 
    neighbors : 9 x pixels in masks, 
    centers : mask centers, 
    isneighbor : valid neighbor boolean 9 x pixels
    """
    nimg = neighbors.shape[0] // 9
    pt = torch.from_numpy(neighbors).to(device)
    
    T = torch.zeros((nimg,Ly,Lx), dtype=torch.double, device=device)
    meds = torch.from_numpy(centers.astype(int)).to(device).long()
    isneigh = torch.from_numpy(isneighbor).to(device)
    with torch.no_grad():
        for i in range(n_iter):
            T[:, meds[:,0], meds[:,1]] +=1
            Tneigh = T[:, pt[:,:,0], pt[:,:,1]]
            Tneigh *= isneigh
            T[:, pt[0,:,0], pt[0,:,1]] = Tneigh.mean(axis=1)
        
        T = torch.log(1.+ T)
        # gradient positions
        grads = T[:, pt[[2,1,4,3],:,0], pt[[2,1,4,3],:,1]]
        dy = grads[:,0] - grads[:,1]
        dx = grads[:,2] - grads[:,3]

        mu_torch = np.stack((dy.cpu().squeeze(), dx.cpu().squeeze()), axis=-2)

    return mu_torch


def masks_to_flows(masks, device):
    # Adapted from https://github.com/MouseLand/cellpose
    """ convert masks to flows using diffusion from center pixel
    Center of masks where diffusion starts is defined to be the 
    closest pixel to the median of all pixels that is inside the 
    mask. Result of diffusion is converted into flows by computing
    the gradients of the diffusion density map. 
    Parameters
    -------------
    masks: int, 2D array
        labelled masks 0=NO masks; 1,2,...=mask labels
    Returns
    -------------
    mu: float, 2D array 
        flows in Y = mu[-2], flows in X = mu[-1].
    mu_c: float, 2D array
        for each pixel, the distance to the center of the mask in which it resides 
    """
    Ly0,Lx0 = masks.shape
    Ly, Lx = Ly0+2, Lx0+2
    masks_padded = np.zeros((Ly, Lx), np.int64)
    masks_padded[1:-1, 1:-1] = masks
    # get mask pixel neighbors
    y, x = np.nonzero(masks_padded)
    neighborsY = np.stack((y, y-1, y+1, 
                           y, y, y-1, 
                           y-1, y+1, y+1), axis=0)
    neighborsX = np.stack((x, x, x, 
                           x-1, x+1, x-1, 
                           x+1, x-1, x+1), axis=0)

    neighbors = np.stack((neighborsY, neighborsX), axis=-1)

    # get mask centers
    slices = scipy.ndimage.find_objects(masks)
    
    centers = np.zeros((masks.max(), 2), 'int')
    for i,si in enumerate(slices):
        if si is not None:
            sr,sc = si
            # ly, lx = sr.stop - sr.start + 1, sc.stop - sc.start + 1
            yi,xi = np.nonzero(masks[sr, sc] == (i+1))
            yi = yi.astype(np.int32) + 1 # add padding
            xi = xi.astype(np.int32) + 1 # add padding
            ymed = np.median(yi)
            xmed = np.median(xi)
            imin = np.argmin((xi-xmed)**2 + (yi-ymed)**2)
            xmed = xi[imin]
            ymed = yi[imin]
            centers[i,0] = ymed + sr.start 
            centers[i,1] = xmed + sc.start

    # get neighbor validator (not all neighbors are in same mask)
    neighbor_masks = masks_padded[neighbors[:,:,0], neighbors[:,:,1]]
    isneighbor = neighbor_masks == neighbor_masks[0]
    ext = []
    for slice_data in slices:
        if slice_data is not None:
            sr, sc = slice_data
            ext.append([sr.stop - sr.start + 1, sc.stop - sc.start + 1])
    ext = np.array(ext)
    n_iter = 2 * (ext.sum(axis=1)).max()
    # run diffusion
    mu = extend_centers(neighbors, centers, isneighbor, Ly, Lx, n_iter=n_iter, device=device)

    # normalize
    mu /= (1e-20 + (mu**2).sum(axis=0)**0.5)

    # put into original image
    mu0 = np.zeros((2, Ly0, Lx0))
    mu0[:, y-1, x-1] = mu
    mu_c = np.zeros_like(mu0)
    return mu0, mu_c,centers


def seg2diffgrads(label,topt):
    # input: (y, x) for 2D data & (z, y, x) z>1 for 3D data
    # output: (3, y, x) for 2D data & (3, z, y, x) for 3D data
    # flows[i][0] is Y flow, flows[i][1] is X flow, flows[i][2] is probability map
    # print('label.shape',label.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    masks = label.squeeze().astype(np.int32)
    # label = np.expand_dims(fastremap.renumber(label, in_place=True)[0],axis=0)
    label = np.expand_dims(label,axis=0)

    if masks.ndim==3:
        z, y, x = masks.shape
        mu = np.zeros((z, 2, y, x), np.float32)
        for z in range(z):
            mu0 = masks_to_flows(masks[z], device=device)[0]
            mu[z] = mu0 
        # for y in trange(y):
        #     mu0 = masks_to_flows(masks[:,y], device=device)[0]
        #     mu[[0,2], :, y] += mu0
        # for x in trange(x):
        #     mu0 = masks_to_flows(masks[:,:,x], device=device)[0]
        #     mu[[0,1], :, :, x] += mu0
        mu = mu.transpose(1,0,2,3)  # mu.shape = (2,z,y,x)
        # concatenate vector flows & cell probability binary maps
        flows = np.concatenate( (mu, label>0.5), axis=0).astype(np.float32)
    elif masks.ndim==2:
        # concatenate vector flows & cell probability binary maps
        mu, mu_c,centers = masks_to_flows(masks, device)
        # mu = np.expand_dims(mu, axis=1)     # mu.shape = (2,1,y,x)
        # label = np.expand_dims(label,axis=0)
        flows = np.concatenate( ( mu, label>0.5 ), axis=0).astype(np.float32)
    else:
        raise ValueError('expected only takes 2D or 3D labels in seg2diffgrads')

    return flows


def seg_to_targets(label_orig: np.ndarray,
                   topts: List[str],
                   erosion_rates: RATES_TYPE = None,
                   dilation_rates: RATES_TYPE = None):
    # input: (D, H, W), output: (C, D, H, W)
    out = [None]*len(topts)

    for tid, topt in enumerate(topts):
        label = label_orig.copy()
        label = erode_label(label, tid, erosion_rates)
        label = dilate_label(label, tid, dilation_rates)

        if topt[0] == '0':  # binary mask
            fg_mask = seg2binary(label, topt)
            out[tid] = fg_mask[np.newaxis, :].astype(np.float32)
        elif topt[0] == '1':  # synaptic polarity
            out[tid] = seg2polarity(label)
        elif topt[0] == '2':  # affinity
            out[tid] = seg2affinity(label, topt)
        elif topt[0] == '3':  # small object mask
            # size_thres: 2d threshold for small size
            # zratio: resolution ration between z and x/y
            # mask_dsize: mask dilation size
            _, size_thres, zratio, _ = [int(x) for x in topt.split('-')]
            out[tid] = (seg_to_small_seg(label, size_thres, zratio) > 0)[
                None, :].astype(np.float32)
        elif topt[0] == '4':  # instance boundary mask
            _, bd_sz, do_bg = [int(x) for x in topt.split('-')]
            if label.ndim == 2:
                out[tid] = seg_to_instance_bd(
                    label[None, :], bd_sz, do_bg).astype(np.float32)
            else:
                out[tid] = seg_to_instance_bd(label, bd_sz, do_bg)[
                    None, :].astype(np.float32)
        elif topt[0] == '5':  # distance transform (instance)
            distance = seg2inst_edt(label, topt)
            out[tid] = distance[np.newaxis, :].astype(np.float32)
        elif topt[0] == '6':  # distance transform (semantic)
            if len(topt) == 1:
                topt = topt + '-2d-8-50'
            assert len(topt.split('-')) == 4
            _, mode, a, b = topt.split('-')
            distance = edt_semantic(label.copy(), mode, float(a), float(b))
            out[tid] = distance[np.newaxis, :].astype(np.float32)
        elif topt[0] == '7':  # cellpose targets (diffusion gradients)
            out[tid] = seg2diffgrads(label,topt)
        elif topt[0] == '9':  # generic semantic segmentation
            out[tid] = label.astype(np.int64)
        else:
            raise NameError("Target option %s is not valid!" % topt[0])

    return out


def syn_sem2inst(label: np.array):
    # For synaptic polarity, convert semantic annotation to instance
    # annotation. It assumes the pre- and post-synaptic masks are
    # closely in touch with their parteners.
    indices = np.unique(label)
    assert list(indices) == [0,1,2]

    fg = (label!=0).astype(bool)
    struct = disk(2, dtype=bool)[np.newaxis,:,:] # only for xy plane
    fg = binary_dilation(fg, struct)
    segm = label_cc(fg).astype(int)

    seg_pos = (label==1).astype(segm.dtype)
    seg_neg = (label==2).astype(segm.dtype)

    seg_pos = seg_pos * (segm * 2 - 1)
    seg_neg = seg_neg * (segm * 2)
    instance_label = np.maximum(seg_pos, seg_neg)

    # Cast the mask to the best dtype to save storage.
    max_id = np.amax(np.unique(instance_label))
    m_type = getSegType(int(max_id))
    return instance_label.astype(m_type)
