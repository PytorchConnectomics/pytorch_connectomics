from __future__ import print_function, division
from typing import Optional, Tuple, List, Union

import numpy as np

def build_blending_matrix(sz, mode='gaussian'):
    assert mode in ['gaussian', 'bump']
    if mode=='gaussian':
        return blend_gaussian(sz)
    else:
        return blend_bump(sz)

def blend_gaussian(sz: Union[Tuple[int], List[int]], 
                   sigma: float=0.2, 
                   mu: float=0.0) -> np.ndarray:  
    """
    Gaussian blending matrix for sliding-window inference.

    Args:
        sz: size of the blending matrix
        sigma (float): standard deviation of the Gaussian distribution. Default: 0.2
        mu (float): mean of the Gaussian distribution. Default: 0.0
    """
    zz, yy, xx = np.meshgrid(np.linspace(-1,1,sz[0], dtype=np.float32), 
                             np.linspace(-1,1,sz[1], dtype=np.float32),
                             np.linspace(-1,1,sz[2], dtype=np.float32), 
                             indexing='ij')

    dd = np.sqrt(zz*zz + yy*yy + xx*xx)
    ww = 1e-4 + np.exp(-( (dd-mu)**2 / ( 2.0 * sigma**2 )))
    return ww.astype(np.float32)

def blend_bump(sz: Union[Tuple[int], List[int]], 
               t: float=1.5) -> np.ndarray: 
    """
    Bump blending matrix for sliding-window inferences (based on [Lee2017]_ with modifications).

    .. [Lee2017] Lee, Kisuk, Jonathan Zung, Peter Li, Viren Jain, and 
        H. Sebastian Seung. "Superhuman accuracy on the SNEMI3D connectomics 
        challenge." arXiv preprint arXiv:1706.00120, 2017.

    Args:
        sz: size of the blending matrix
        t: parameter of the bump blending function. Default: 1.5
    """
    zz, yy, xx = np.meshgrid(np.linspace(0,1,sz[0]+2, dtype=np.float32)[1:-1],
                             np.linspace(0,1,sz[1]+2, dtype=np.float32)[1:-1],
                             np.linspace(0,1,sz[2]+2, dtype=np.float32)[1:-1], 
                             indexing='ij')

    dd = -(xx*(1-xx))**(-t)-(yy*(1-yy))**(-t)-(zz*(1-zz))**(-t)
    ww = 1e-4 + np.exp(dd-dd.max())
    return ww.astype(np.float32)
