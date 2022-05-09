from __future__ import print_function, division
from typing import Optional, List
from collections import OrderedDict

import numpy as np
import itertools
import torch
from skimage.transform import resize
from connectomics.model.utils import SplitActivation


def _forward(model, volume):
    output = model(volume)
    assert isinstance(output, (torch.Tensor, OrderedDict)), "Output is expected to be " + \
        f"torch.Tensor or OrderedDict, but got {type(output)}!"
    if isinstance(output, torch.Tensor):
        return output

    main_key = list(output.keys())[0]
    return output[main_key]


class TestAugmentor(object):
    r"""Test-time spatial augmentor. 

    Our test-time augmentation includes horizontal/vertical flips over 
    the `xy`-plane, swap of `x` and `y` axes, and flip in `z`-dimension, 
    resulting in 16 variants. Considering inference efficiency, we also 
    provide the option to apply only horizontal/vertical flips over the 
    `xy`-plane, resulting in 4 variants. The augmentation can also be applied 
    to 2D outputs without the `z`-flip. By default the test-time augmentor 
    returns the pixel-wise mean value of the predictions.

    Args:
        mode (str): one of ``'min'``, ``'max'`` or ``'mean'``. Default: ``'mean'``
        do_2d (bool): the test-time augmentation is applied to 2d images. Default: False
        num_aug (int, optional): number of data augmentation variants: 4, 8 or 16 (3D only). Default: None
        scale_factors (List[float]): scale factors for resizing the model output. Default: [1.0, 1.0, 1.0]

    Examples::
        >>> from connectomics.data.augmentation import TestAugmentor
        >>> test_augmentor = TestAugmentor(mode='mean', num_aug=16)
        >>> output = test_augmentor(model, inputs) # output is a numpy.ndarray on CPU
    """

    def __init__(self,
                 mode: str = 'mean',
                 do_2d: bool = False,
                 num_aug: Optional[int] = None,
                 scale_factors: List[float] = [1.0, 1.0, 1.0],
                 inference_act=None):
        assert mode in ['mean', 'max', 'min']
        self.mode = mode
        self.do_2d = do_2d
        self.scale_factors = scale_factors
        self.inference_act = inference_act

        if num_aug is not None:
            assert num_aug in [4, 8, 16], \
                "TestAugmentor.num_aug should be either 4, 8 or 16!"
            if self.do_2d: # max num_aug for 2d images
                num_aug = min(num_aug, 8)

        self.num_aug = num_aug

    def __call__(self, model, data):
        if self.do_2d:
            assert len(data.shape) == 4, \
                "The input has a shape of {}, which not a valid 2D " \
                "input in (B, C, H, W) format.".format(data.shape)
            return self._tta_2d(model, data)
        else:
            assert len(data.shape) == 5, \
                "The input has a shape of {}, which not a valid 3D " \
                "input in (B, C, Z, Y, X) format.".format(data.shape)
            return self._tta_3d(model, data)

    def _tta_3d(self, model, data):
        # output in (B, C, Z, Y, X) format
        out = None
        cc = 0

        if self.num_aug == None:
            opts = itertools.product(
                (False, ), (False, ), (False, ), (False, ))
        elif self.num_aug == 4:
            opts = itertools.product(
                (False, True), (False, True), (False, ), (False, ))
        elif self.num_aug == 8:
            opts = itertools.product(
                (False, True), (False, True), (False, ), (False, True))
        else:
            opts = itertools.product(
                (False, True), (False, True), (False, True), (False, True))

        for xflip, yflip, zflip, transpose in opts:
            volume = data.clone()

            if xflip:
                volume = torch.flip(volume, [4])
            if yflip:
                volume = torch.flip(volume, [3])
            if zflip:
                volume = torch.flip(volume, [2])
            if transpose:
                volume = torch.transpose(volume, 3, 4)

            if self.inference_act is not None:
                vout = self.inference_act(
                    _forward(model, volume)).detach().cpu()
            else:
                vout = model(_forward(model, volume)).detach().cpu()

            if transpose:  # swap x-/y-axis
                vout = torch.transpose(vout, 3, 4)
            if zflip:
                vout = torch.flip(vout, [2])
            if yflip:
                vout = torch.flip(vout, [3])
            if xflip:
                vout = torch.flip(vout, [4])

            out = self._update_output(vout, out)
            cc += 1

        if self.mode == 'mean':
            out = out/cc

        if (np.array(self.scale_factors) != 1).any():
            sf = [1.0, 1.0] + self.scale_factors
            spatial_size = np.array(out.shape) * np.array(sf)
            spatial_size = list(np.ceil(spatial_size).astype(int))
            out = resize(out, spatial_size, order=1, preserve_range=True,
                         anti_aliasing=True)
        return out

    def _tta_2d(self, model, data):
        # output in (B, C, Y, X) format
        out = None
        cc = 0

        if self.num_aug == None:
            opts = itertools.product((False, ), (False, ), (False, ))
        elif self.num_aug == 4:
            opts = itertools.product((False, True), (False, True), (False, ))
        else:
            opts = itertools.product(
                (False, True), (False, True), (False, True))

        for xflip, yflip, transpose in opts:
            volume = data.clone()

            if xflip:
                volume = torch.flip(volume, [3])
            if yflip:
                volume = torch.flip(volume, [2])
            if transpose:
                volume = torch.transpose(volume, 2, 3)

            if self.inference_act is not None:
                vout = self.inference_act(
                    _forward(model, volume)).detach().cpu()
            else:
                vout = _forward(model, volume).detach().cpu()

            if transpose:  # swap x-/y-axis
                vout = torch.transpose(vout, 2, 3)
            if yflip:
                vout = torch.flip(vout, [2])
            if xflip:
                vout = torch.flip(vout, [3])

            out = self._update_output(vout, out)
            cc += 1

        if self.mode == 'mean':
            out = out/cc

        if (np.array(self.scale_factors)[1:] != 1).any():
            sf = [1.0, 1.0] + self.scale_factors[1:]
            spatial_size = np.array(out.shape) * np.array(sf)
            spatial_size = list(np.ceil(spatial_size).astype(int))
            out = resize(out, spatial_size, order=1, preserve_range=True,
                         anti_aliasing=True)
        return out

    def _update_output(self, vout, out=None):
        # cast to numpy array
        vout = vout.numpy()
        if out is None:
            if self.mode == 'min':
                out = np.ones(vout.shape, dtype=np.float32)
            elif self.mode == 'max':
                out = np.zeros(vout.shape, dtype=np.float32)
            elif self.mode == 'mean':
                out = np.zeros(vout.shape, dtype=np.float32)

        if self.mode == 'min':
            out = np.minimum(out, vout)
        elif self.mode == 'max':
            out = np.maximum(out, vout)
        elif self.mode == 'mean':
            out += vout

        return out

    def update_name(self, name):
        r"""Update the name of the output file to indicate applied test-time augmentations.
        """
        extension = "_"
        if self.num_aug is None:
            return name
        elif self.num_aug == 4:
            extension += "xy"
        elif self.num_aug == 8:
            extension += "txy"
        elif self.num_aug == 16:
            extension += "txyz"

        # Update the suffix of the output filename to indicate
        # the use of test-time data augmentation.
        name_list = name.split('.')
        new_filename = name_list[0] + extension + "." + name_list[1]
        return new_filename

    @classmethod
    def build_from_cfg(cls, cfg, activation=True):
        r"""Build a TestAugmentor from configs.
        """
        act = None
        if activation:
            act = SplitActivation.build_from_cfg(cfg, normalize=True)

        return cls(mode=cfg.INFERENCE.AUG_MODE,
                   do_2d=cfg.DATASET.DO_2D,
                   num_aug=cfg.INFERENCE.AUG_NUM,
                   scale_factors=cfg.INFERENCE.OUTPUT_SCALE,
                   inference_act=act)
