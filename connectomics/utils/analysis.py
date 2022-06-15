from __future__ import print_function, division

import numpy as np  # handle volume data
import pandas as pd  # use pandas data frames for plotting with seaborn
from tqdm import tqdm  # show progress
# derive the center of mass of instances
from scipy.ndimage.measurements import center_of_mass
from scipy.spatial import KDTree

from connectomics.data.utils import *


def voxel_instance_size(target: np.ndarray, ds_name: str = 'main') -> pd.DataFrame:
    ''' Calculate the voxel based size of each instance in an instance segmentation map.

        Args:
            target: The target data as numpy ndarray
            ds_name: Name of the dataset, saved in pd column

        Return
            A single column Panda data frame that contains the voxel based instance sizes.
    '''

    # find the unique values and their number of accurancy
    idx, count = np.unique(target, return_counts=True)

    # save the pixel count to a pandas data frame
    idx_pix_count = {x: y for x, y in zip(
        idx[1:], count[1:])}  # 1:, skip background
    idx_pix_count_pd = pd.DataFrame(data=list(idx_pix_count.values()), columns=["Size"], 
                                    index=list(idx_pix_count.keys()))
    idx_pix_count_pd["Dataset"] = ds_name

    return idx_pix_count_pd


def distance_nn(target: np.ndarray, ds_name: str = 'main', 
                resolution=[1.0, 1.0, 1.0]) -> pd.DataFrame:
    ''' Caculate the distance to the NN for each instance in the target matrix. 

        Args:
            target: The target data as numpy ndarray
            ds_name: Name of the dataset, saved in pd column
            resolution: Axis scaling factors in case of anisotropy

        Return
            A single column Panda data frame that contains the distance of each instance to its NN
    '''
    # convert the instance map to binary
    binary = (target != 0).astype(np.uint8)

    # derive the center of mass of each instance in the target matrix
    cm = center_of_mass(binary, target, list(np.unique(target))[1:])
    cm = np.array(cm) * np.array(resolution)[None, :]

    kd_tree = KDTree(cm)
    # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html
    distance, _ = kd_tree.query(cm, k=2)
    distance = np.array(distance)[:,1]

    # write the distance value to a pandas data frame
    idx_zxy_values_pd = pd.DataFrame(data=list(distance), columns=["NN_Distance"])
    idx_zxy_values_pd["Dataset"] = ds_name

    return idx_zxy_values_pd


def pixel_intensity(source: np.ndarray, target: np.ndarray, bOrF: str = 'foreground', 
                    ds_name: str = 'main') -> pd.DataFrame:
    ''' Retrives the intesity of each pixel. Writes them to a Pandas data frame.
        Can handle background for foreground.
        Args:
            source: Source numpy ndarray
            target: Target numpy ndarray
            bOrF: Either 'foreground' or 'background', indicates which intensities to estimate
            ds_name: Name of the dataset, saved in pd column

        Return
            A pandas frame with the intensity of each pixel, if the pixel belongs to the background
            or foreground, and the dataset it belongs to.
    '''

    # mask out forground or background
    assert bOrF in ['foreground', 'background'], \
        f"bOrF has to be \"foreground\" or \"background\", not {bOrF}"
    mask_bOrF = 1 if bOrF == 'foreground' else 0

    mask = target > 0
    masked_source = source[mask == mask_bOrF]

    # create the pd data frame
    pix_int_count_front_pd = pi_pd(masked_source, bOrF, ds_name)

    return pix_int_count_front_pd


def pi_pd(mask: np.ndarray, bOrF: str = 'foreground', ds_name: str = 'main') -> pd.DataFrame:
    ''' Creates pandas data frame of the intesity of all pixels in the mask.
        Used by: pixel_intensity()

        Args:
            mask: Numpy array of masked out pixels 
            bOrF: Either 'foreground' or 'background', indicates which intesities are estimated
            ds_name: Name of the dataset

        Return
            Pandas data frame of the intesities of each pixel in the mask array

    '''
    # convert masked matrix to 1D array and write to pd
    mask_1D = mask.ravel()
    pix_int_count_pd = pd.DataFrame(data=mask_1D, columns=["Intensity"])

    # add column with back or forground specification
    pix_int_count_pd["B/F"] = bOrF

    # add column with dataset name
    pix_int_count_pd["Dataset"] = ds_name

    return pix_int_count_pd


def diff_segm(seg1: np.ndarray, seg2: np.ndarray, iou_thres: float = 0.75, 
              progress: bool = False) -> dict:
    """Check the differences between two 3D instance segmentation maps. The 
    background pixels (value=0) are ignored.

    Args:
        seg1 (np.ndarray): the first segmentation map.
        seg2 (np.ndarray): the second segmentation map.
        iou_thres (float): the threshold of intersection-over-union. Default: 0.75
        progress (bool): show progress bar. Default: False

    Returns:
        dict: a dict contains lists of shared and unique indicies

    Note:
        The shared segments in two segmentation maps can have different indices,
        therefore they are saved separately in the output dict.
    """
    def _get_indices_counts(seg: np.ndarray):
        # return indices and counts while ignoring the background
        indices, counts = np.unique(seg, return_counts=True)
        if indices[0] == 0:
            return indices[1:], counts[1:]
        else:
            return indices, counts

    results ={
        "seg1_unique": [],
        "seg2_unique": [],
        "shared1": [],
        "shared2": [],
    }

    indices1, counts1 = _get_indices_counts(seg1)
    indices2, counts2 = _get_indices_counts(seg2)
    if len(indices1) == 0: # no non-background objects
        results["seg2_unique"] = list(indices2)
        return results
    if len(indices2) == 0:
        results["seg1_unique"] = list(indices1)
        return results
    
    counts_dict1 = dict(zip(indices1, counts1))
    counts_dict2 = dict(zip(indices2, counts2))
    bbox_dict1 = index2bbox(seg1, indices1, relax=1, progress=progress)

    for idx1 in (tqdm(indices1) if progress else indices1):
        bbox = bbox_dict1[idx1]
        crop_seg1, crop_seg2 = crop_ND(seg1, bbox), crop_ND(seg2, bbox)
        temp1 = (crop_seg1==idx1).astype(int)

        best_iou = 0.0
        crop_indices = np.unique(crop_seg2)
        for idx2 in crop_indices:
            if idx2 == 0: # ignore background
                continue 
            temp2 = (crop_seg2==idx2).astype(int)
            overlap = (temp1*temp2).sum()
            union = counts_dict1[idx1] + counts_dict2[idx2] - overlap
            iou = overlap / float(union)
            if iou > best_iou:
                best_iou = iou
                matched_idx2 = idx2

        if best_iou < iou_thres:
            results["seg1_unique"].append(idx1)
        else: # the segment is shared in both segmentation maps
            results["shared1"].append(idx1)
            results["shared2"].append(matched_idx2)

    # "seg2_unique" contains elements in indices2 but not in "shared2"
    results["seg2_unique"] = list(set(indices2) - set(results["shared2"]))
    return results
