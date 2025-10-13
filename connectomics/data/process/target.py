from __future__ import print_function, division
from typing import Optional, Union, List

import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
from skimage.morphology import erosion, dilation, disk
import cc3d
import fastremap
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter

from .blend import *
from .distance import *
from .flow import seg2d_to_flows
from .distance import edt_instance, edt_semantic
from .segment import im_to_col

RATES_TYPE = Optional[Union[List[int], int]]

__all__ = [
    'seg_to_flows',
    'seg_to_affinity',
    'seg_to_polarity',
    'seg_to_small_seg',
    'seg_to_binary',
    'seg_to_instance_bd',
    'seg_to_instance_edt',
    'seg_to_semantic_edt',
    'seg_to_generic_semantic',
    'seg_erosion_dilation',
]

def seg_to_flows(label: np.ndarray) -> np.array:
    # input: (y, x) for 2D data or (z, y, x) with z>1 for 3D data
    # output: (2, y, x) for 2D data & (2, z, y, x) for 3D data (channel first)
    masks = label.squeeze().astype(np.int32)

    if masks.ndim==3:
        z, y, x = masks.shape
        mu = np.zeros((2, z, y, x), np.float32)
        for z in range(z):
            mu0 = seg2d_to_flows(masks[z])[0]
            mu[:, z] = mu0 
        flows = mu.astype(np.float32)
    elif masks.ndim==2:
        mu, _, _ = seg2d_to_flows(masks)
        flows = mu.astype(np.float32)
    else:
        raise ValueError('expecting 2D or 3D labels but received %dD input!' % masks.ndim)

    return flows


def seg_to_instance_bd(seg: np.ndarray,
                       thickness: int = 1,
                       do_bg: bool = True,
                       do_convolve: bool = True) -> np.ndarray:
    """Generate instance contour map from segmentation masks.

    Args:
        seg (np.ndarray): segmentation map (3D array is required).
        thickness (int, optional): thickness of the boundary (half-size of dilation struct). Defaults: 1
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
    tsz = thickness*2+1

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
        patch = im_to_col(
            np.pad(seg[z], ((thickness, thickness), (thickness, thickness)), 'reflect'), [tsz, tsz])
        p0 = patch.max(axis=1)
        if do_bg:  # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0 > 0)*(p0 != p1)).reshape(sz[1:])
        else:  # between two non-zero seg
            patch[patch == 0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0 != 0)*(p1 != 0)*(p0 != p1)).reshape(sz[1:])
    return bd


def seg_to_binary(label, segment_id=[]):
    """
    Convert segmentation to binary mask.
    
    Args:
        label: Segmentation array
        segment_id: List of segment IDs to include as foreground.
                   If empty list [], returns all non-zero labels.
                   
    Returns:
        Binary mask where specified segments are foreground
    """
    # If empty list, return all non-zero labels
    if not segment_id:
        return label > 0
    
    # Create foreground mask for specified segment IDs
    fg_mask = np.zeros_like(label).astype(bool)
    for seg_id in segment_id:
        fg_mask = np.logical_or(fg_mask, label == int(seg_id))
    return fg_mask


def seg_to_affinity(seg: np.ndarray, target_opt: List[str]) -> np.ndarray:
    """
    Compute affinities from a segmentation based on target options.

    Args:
        seg: The segmentation to compute affinities from. Shape: (z, y, x).
        target_opt: List of strings defining affinity offsets.
            Can be either:
            - Legacy format: ['1', '0-0-1', '0-1-0', ...] (first element is type indicator)
            - Modern format: ['0-0-1', '0-1-0', ...] (direct offset list)

    Returns:
        The affinities. Shape: (num_offsets, z, y, x).
    """
    if len(target_opt) == 0:
        # Default short-range affinities
        offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        # Detect format: check if first element is a type indicator or an offset
        start_idx = 0
        if len(target_opt) > 0 and '-' not in target_opt[0]:
            # Legacy format: first element is type indicator (e.g., '1')
            start_idx = 1
        
        # Parse offsets from target_opt
        offsets = []
        for opt_str in target_opt[start_idx:]:
            if '-' in opt_str:
                offset = [int(x) for x in opt_str.split('-')]
                offsets.append(offset)
        
        # Fallback to default if no valid offsets found
        if len(offsets) == 0:
            offsets = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    num_offsets = len(offsets)
    affinities = np.zeros((num_offsets, *seg.shape), dtype=np.float32)

    for i, offset in enumerate(offsets):
        dz, dy, dx = offset

        # Create slices for the offset
        if dz > 0:
            src_slice = (slice(None, -dz), slice(None), slice(None))
            dst_slice = (slice(dz, None), slice(None), slice(None))
        elif dz < 0:
            src_slice = (slice(-dz, None), slice(None), slice(None))
            dst_slice = (slice(None, dz), slice(None), slice(None))
        else:
            src_slice = (slice(None), slice(None), slice(None))
            dst_slice = (slice(None), slice(None), slice(None))

        if dy > 0:
            src_slice = (src_slice[0], slice(None, -dy), src_slice[2])
            dst_slice = (dst_slice[0], slice(dy, None), dst_slice[2])
        elif dy < 0:
            src_slice = (src_slice[0], slice(-dy, None), src_slice[2])
            dst_slice = (dst_slice[0], slice(None, dy), dst_slice[2])

        if dx > 0:
            src_slice = (src_slice[0], src_slice[1], slice(None, -dx))
            dst_slice = (dst_slice[0], dst_slice[1], slice(dx, None))
        elif dx < 0:
            src_slice = (src_slice[0], src_slice[1], slice(-dx, None))
            dst_slice = (dst_slice[0], dst_slice[1], slice(None, dx))

        # Compute affinity
        affinities[i][dst_slice] = (seg[src_slice] == seg[dst_slice]) & (seg[dst_slice] > 0)

    return affinities


def seg_to_polarity(label: np.ndarray, exclusive: bool = False) -> np.ndarray:
    """Convert the label to synaptic polarity target.
    
    Args:
        label: Segmentation array where odd labels are pre-synaptic, even are post-synaptic
        exclusive: If False, returns 3-channel non-exclusive masks (for BCE loss).
                  If True, returns single-channel exclusive classes (for CE loss).
    
    Returns:
        Polarity masks: 3 channels (pre, post, all) if exclusive=False,
                       or 1 channel (0=bg, 1=pre, 2=post) if exclusive=True
    """
    pos = np.logical_and((label % 2) == 1, label > 0)
    neg = np.logical_and((label % 2) == 0, label > 0)

    if not exclusive:
        # Convert segmentation to 3-channel synaptic polarity masks.
        # The three channels are not exclusive. They are learned by 
        # binary cross-entropy (BCE) losses after per-pixel sigmoid.
        tmp = [None]*3
        tmp[0], tmp[1], tmp[2] = pos, neg, (label > 0)
        return np.stack(tmp, 0).astype(np.float32)

    # Learn the exclusive semantic (synaptic polarity) masks
    # using the cross-entropy (CE) loss for three classes.
    return np.maximum(pos.astype(np.int64), 2*neg.astype(np.int64))


def seg_to_synapse_instance(label: np.array):
    # For synaptic polarity, convert semantic annotation to instance
    # annotation. It assumes the pre- and post-synaptic masks are
    # closely in touch with their parteners.
    indices = np.unique(label)
    assert list(indices) == [0,1,2]

    fg = (label!=0).astype(bool)
    struct = disk(2, dtype=bool)[np.newaxis,:,:] # only for xy plane
    fg = binary_dilation(fg, struct)
    segm = cc3d.connected_components(fg).astype(int)

    seg_pos = (label==1).astype(segm.dtype)
    seg_neg = (label==2).astype(segm.dtype)

    seg_pos = seg_pos * (segm * 2 - 1)
    seg_neg = seg_neg * (segm * 2)
    instance_label = np.maximum(seg_pos, seg_neg)

    # Cast the mask to the best dtype to save storage.
    return fastremap.refit(instance_label)


def seg_to_small_seg(seg: np.ndarray, threshold: int = 100) -> np.ndarray:
    """Convert segmentation to small object mask.

    Args:
        seg: Input segmentation array
        threshold: Maximum voxel count for objects to be considered small (default: 100)

    Returns:
        Small object mask (1.0 for small objects, 0.0 otherwise)
    """
    # Use connected components to find small objects
    labeled_seg = cc3d.connected_components(seg)
    unique_labels, counts = np.unique(labeled_seg, return_counts=True)

    # Create mask for small objects
    small_mask = np.zeros_like(seg, dtype=np.float32)
    for label, count in zip(unique_labels, counts):
        if count <= threshold and label > 0:  # exclude background
            small_mask[labeled_seg == label] = 1.0

    return small_mask




def seg_to_generic_semantic(seg: np.ndarray, class_ids: List[int] = []) -> np.ndarray:
    """Convert segmentation to generic semantic mask.

    Args:
        seg: Input segmentation array
        class_ids: List of class IDs to map to semantic classes.
                  If empty, returns binary (foreground vs background).
                  Otherwise, maps each class_id to semantic class 1, 2, 3, ...

    Returns:
        Generic semantic mask
    """
    if not class_ids:
        # Simple binary semantic: foreground vs background
        return (seg > 0).astype(np.float32)

    # Multi-class semantic based on class_ids
    result = np.zeros_like(seg, dtype=np.float32)
    for i, class_id in enumerate(class_ids, 1):
        result[seg == class_id] = i

    return result


def seg_to_instance_edt(seg: np.ndarray, mode: str = '2d', quantize: bool = False) -> np.ndarray:
    """Convert segmentation to instance EDT.

    Args:
        seg: Input segmentation array
        mode: EDT computation mode: '2d' or '3d' (default: '2d')
        quantize: Whether to quantize the EDT values (default: False)

    Returns:
        Instance EDT array
    """
    # Set appropriate resolution based on mode
    if mode == '2d':
        resolution = (1.0, 1.0)  # 2D resolution
    else:
        resolution = (1.0, 1.0, 1.0)  # 3D resolution
    
    return edt_instance(seg, mode=mode, quantize=quantize, resolution=resolution)


def seg_to_semantic_edt(seg: np.ndarray, 
                        mode: str = '2d',
                        alpha_fore: float = 8.0, 
                        alpha_back: float = 50.0) -> np.ndarray:
    """Convert segmentation to semantic EDT.

    Args:
        seg: Input segmentation array
        mode: EDT computation mode: '2d' or '3d' (default: '2d')
        alpha_fore: Foreground distance weight (default: 8.0)
        alpha_back: Background distance weight (default: 50.0)

    Returns:
        Semantic EDT array
    """
    return edt_semantic(seg, mode=mode, alpha_fore=alpha_fore, alpha_back=alpha_back)


def seg_erosion_dilation(seg: np.ndarray, 
                         operation: str = 'erosion',
                         kernel_size: int = 1) -> np.ndarray:
    """Apply erosion and/or dilation to segmentation.

    Args:
        seg: Input segmentation array
        operation: Operation type: 'erosion', 'dilation', or 'both' (default: 'erosion')
        kernel_size: Kernel size for morphological operation (default: 1)

    Returns:
        Processed segmentation
    """
    # Create structuring element
    struct_elem = disk(kernel_size, dtype=bool)
    if seg.ndim == 3:
        struct_elem = struct_elem[np.newaxis, :, :]

    result = seg.copy()

    if operation == 'erosion':
        for z in range(seg.shape[0]):
            result[z] = erosion(seg[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
    elif operation == 'dilation':
        for z in range(seg.shape[0]):
            result[z] = dilation(seg[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
    elif operation == 'both':
        # First erosion
        for z in range(seg.shape[0]):
            result[z] = erosion(seg[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
        # Then dilation
        for z in range(seg.shape[0]):
            result[z] = dilation(result[z], struct_elem[0] if seg.ndim == 3 else struct_elem)
    else:
        raise ValueError(f"Unknown operation: {operation}. Use 'erosion', 'dilation', or 'both'")

    return result
