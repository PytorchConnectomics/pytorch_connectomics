from __future__ import print_function, division
from typing import Optional, List

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .block import *
from .utils import get_functional_act

class ResNet3D(nn.Module):
    """ResNet backbone for 3D semantic/instance segmentation. 
       The global average pooling and fully-connected layer are removed.
    """
    def __init__(self):
        pass