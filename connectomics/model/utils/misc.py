from __future__ import print_function, division
from collections import OrderedDict
from typing import Optional, List

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model, adapted
    from https://github.com/pytorch/vision/blob/master/torchvision/models/_utils.py.

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class SplitActivation(object):
    r"""Apply different activation functions for the outpur tensor.
    """
    # number of channels of different target options
    num_channels_dict = {
        '0': 1, # binary foreground
        '1': 3, # synaptic polarity
        '2': 3, # affinity
        '3': 1, # small object
        '4': 1, # instance boundary
        '5': 1, # instance edt (11 channels for quantized)
        '6': 1, # semantic edt
        '7': 2, # diffusion gradients (2d)
        'all': -1 # all remaining channels
    }

    def __init__(self,
                 target_opt: List[str] = ['0'],
                 output_act: Optional[List[str]] = None,
                 split_only: bool = False,
                 do_cat: bool = True,
                 do_2d: bool = False,
                 normalize: bool = False):

        if output_act is not None:
            assert len(target_opt) == len(output_act)
        if do_2d: # 2d affinity only has x and y
            self.num_channels_dict['2'] = 2

        self.split_channels = []
        self.target_opt = target_opt
        self.do_cat = do_cat
        self.normalize = normalize

        self.split_only = split_only
        if not self.split_only:
            self.act = self._get_act(output_act)

        for i, topt in enumerate(self.target_opt):
            assert isinstance(topt, str)
            if i < len(self.target_opt) - 1:
                assert topt != 'all', "Only last target can be all"

            if topt[0] == 'I': # image with specified channel number
                if len(topt) == 1:
                    topt = topt + '-1' # gray-scale image
                _, channels = topt.split('-')
                self.split_channels.append(int(channels))
                continue

            if topt[0] == '5': # instance_edt
                if len(topt) == 1:
                    topt = topt + '-2d-0-0-5.0' # 2d w/o padding or quantize
                _, mode, padding, quant, z_res = topt.split('-')
                if bool(int(quant)): # quantized by 0.1 bin (additional one for bg)
                    self.split_channels.append(11)
                    continue

            if topt[0] == '9': # semantic masks
                channels = int(topt.split('-')[1])
                self.split_channels.append(channels)
                continue


            # use the default channel number for other cases
            self.split_channels.append(self.num_channels_dict[topt[0]])
        print("Channel split rule for prediction: ", self.split_channels)

    def __call__(self, x):
        split_channels = self.split_channels.copy()
        if split_channels[-1] == -1:
            split_channels[-1] = x.shape[1] - sum(split_channels[:-1])
        x = torch.split(x, split_channels, dim=1)
        x = list(x)  # torch.split returns a tuple
        if self.split_only:
            return x

        x = [self._apply_act(self.act[i], x[i])
             for i in range(len(x))]

        if self.do_cat:
            return torch.cat(x, dim=1)
        return x

    def _get_act(self, act):
        num_target = len(self.target_opt)
        out = [None]*num_target
        for i, act in enumerate(act):
            out[i] = get_functional_act(act)
        return out

    def _apply_act(self, act_fn, x):
        x = act_fn(x)
        if self.normalize and act_fn == torch.tanh:
            x = (x + 1.0) / 2.0

        return x

    @classmethod
    def build_from_cfg(cls,
                       cfg,
                       do_cat: bool = True,
                       split_only: bool = False,
                       normalize: bool = False):

        return cls(cfg.MODEL.TARGET_OPT,
                   cfg.INFERENCE.OUTPUT_ACT,
                   split_only=split_only,
                   do_cat=do_cat,
                   do_2d=cfg.DATASET.DO_2D,
                   normalize=normalize)


class ImagePool(object):
    """This class implements an image buffer that stores previously generated images. Adapted from
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/util/image_pool.py

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """
    def __init__(self, pool_size: int, device: torch.device, on_cpu: bool=False):
        """Initialize the ImagePool class

        Args:
            pool_size (int): the size of image buffer, if pool_size=0, no buffer will be created
            device (torch.device): model running device. GPUs are recommended for model training and inference.
            on_cpu (bool): whether to save image buffer on cpu to reduce GPU memory usage. Defalt: False
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

        self.device, self.on_cpu = device, on_cpu

    def query(self, images):
        """Return an image from the pool.

        Args:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images

        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                if self.on_cpu: # save buffer images on cpu
                    self.images.append(image.clone().detach().cpu())
                else:
                    self.images.append(image.clone())
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    if self.on_cpu: # save buffer images on cpu
                        self.images[random_id] = image.clone().detach().cpu()
                    else:
                        self.images.append(image.clone())
                    return_images.append(tmp.to(self.device))
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


# ------------------
# Swish Activation
# ------------------
# An ordinary implementation of Swish function


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# A memory-efficient implementation of Swish function


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

# --------------------
# Activation Layers
# --------------------


def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer.

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``,
            ``'silu'``, ``'swish'``, 'efficient_swish'`` and ``'none'``. Default: ``'relu'``
    """
    assert activation in ["relu", "leaky_relu", "elu", "gelu", "silu",
                          "swish", "efficient_swish", "none"], \
        "Get unknown activation key {}".format(activation)
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.2, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(inplace=True),
        "swish": Swish(),
        "efficient_swish": MemoryEfficientSwish(),
        "none": nn.Identity(),
    }
    return activation_dict[activation]


def get_functional_act(activation: str = 'relu'):
    """Get the specified activation function.

    Args:
        activation (str): one of ``'relu'``, ``'tanh'``, ``'elu'``, ``'sigmoid'``,
            ``'softmax'`` and ``'none'``. Default: ``'sigmoid'``
    """
    assert activation in ["relu", "tanh", "elu", "sigmoid", "softmax", "none"], \
        "Get unknown activation_fn key {}".format(activation)
    activation_dict = {
        'relu': F.relu_,
        'tanh': torch.tanh,
        'elu': F.elu_,
        'sigmoid': torch.sigmoid,
        'softmax': lambda x: F.softmax(x, dim=1),
        'none': lambda x: x,
    }
    return activation_dict[activation]


# ----------------------
# Normalization Layers
# ----------------------


def get_norm_3d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 3D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    if norm == "gn": assert out_channels%8 == 0, "GN requires channels to separable into 8 groups"
    norm = {
        "bn": nn.BatchNorm3d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm3d,
        "gn": lambda channels: nn.GroupNorm(8, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_norm_2d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 2D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    norm = {
        "bn": nn.BatchNorm2d,
        "sync_bn": nn.SyncBatchNorm,
        "in": nn.InstanceNorm2d,
        "gn": lambda channels: nn.GroupNorm(16, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_norm_1d(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer for a 1D model.

    Args:
        norm (str): one of ``'bn'``, ``'sync_bn'`` ``'in'``, ``'gn'`` or ``'none'``.
        out_channels (int): channel number.
        bn_momentum (float): the momentum of normalization layers.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["bn", "sync_bn", "gn", "in", "none"], \
        "Get unknown normalization layer key {}".format(norm)
    norm = {
        "bn": nn.BatchNorm1d,
        "sync_bn": nn.BatchNorm1d,
        "in": nn.InstanceNorm1d,
        "gn": lambda channels: nn.GroupNorm(16, channels),
        "none": nn.Identity,
    }[norm]
    if norm in ["bn", "sync_bn", "in"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)


def get_num_params(model):
    num_param = sum([param.nelement() for param in model.parameters()])
    return num_param
