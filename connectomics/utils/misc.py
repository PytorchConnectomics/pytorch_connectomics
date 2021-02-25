from __future__ import print_function, division
from typing import Optional, Union, List, Tuple

import itertools
import numpy as np

def bbox_ND(img: np.ndarray, relax: int = 0) -> tuple:
    """Calculate the bounding box of an object in a N-dimensional
    numpy array. All non-zero elements are treated as foregounrd.
    Reference: https://stackoverflow.com/a/31402351

    Args:
        img (np.ndarray): a N-dimensional array with zero as background.

    Returns:
        tuple: N-dimensional bounding box coordinates.
    """
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])

    return bbox_relax(out, img.shape, relax)

def bbox_relax(coord: Union[tuple, list], 
               shape: tuple, 
               relax: int = 0) -> tuple:

    assert len(coord) == len(shape) * 2
    coord = list(coord)
    for i in range(len(shape)):
        coord[2*i] = max(0, coord[2*i]-relax)
        coord[2*i+1] = min(shape[i], coord[2*i+1]+relax)

    return tuple(coord)

def crop_ND(img: np.ndarray, coord: Tuple[int]) -> np.ndarray:
    N = img.ndim
    assert len(coord) == N * 2
    slicing = []
    for i in range(N):
        slicing.append(slice(coord[2*i], coord[2*i+1]))
    slicing = tuple(slicing)
    return img[slicing].copy()
