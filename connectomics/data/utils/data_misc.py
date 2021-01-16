from __future__ import print_function, division
from typing import Optional, Tuple, List, Union
import numpy as np

def get_padsize(pad_size: Union[int, List[int]], ndim: int=3) -> Tuple[int]:
    """Convert the padding size for 3D input volumes into numpy.pad compatible format.

    Args:
        pad_size (int, List[int]): number of values padded to the edges of each axis. 
        ndim (int): the dimension of the array to be padded. Default: 3
    """
    if type(pad_size) == int:
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)

    assert len(pad_size) in [1, ndim, 2*ndim]
    if len(pad_size) == 1:
        pad_size = pad_size[0]
        pad_size = [tuple([pad_size, pad_size]) for _ in range(ndim)]
        return tuple(pad_size)
    elif len(pad_size) == ndim:
        return tuple([tuple([x, x]) for x in pad_size])
    else:
        return tuple(
            [tuple([pad_size[2*i], pad_size[2*i+1]]) 
            for i in range(len(pad_size) // 2)])

def array_unpad(data: np.ndarray,
                pad_size: Tuple[int]) -> np.ndarray:
    """Unpad a given numpy.ndarray based on the given padding size.

    Args:
        data (numpy.ndarray): the input volume to unpad.
        pad_size (tuple): number of values removed from the edges of each axis. 
            Should be in the format of ((before_1, after_1), ... (before_N, after_N)) 
            representing the unique pad widths for each axis.
    """
    diff = data.ndim - len(pad_size)
    if diff > 0:
        extra = [(0, 0) for _ in range(diff)]
        pad_size = tuple(extra + list(pad_size))

    assert len(pad_size) == data.ndim
    index = tuple([
        slice(pad_size[i][0], data.shape[i]-pad_size[i][1]) 
        for i in range(data.ndim)
    ])
    return data[index]
