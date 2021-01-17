from __future__ import print_function, division
from collections import OrderedDict

import torch
from torch import nn
from torch.jit.annotations import Dict


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a 3D model, adapted
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


#------------------
# Swish Activation
#------------------
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

#--------------------
# Activation Layers
#--------------------
def get_activation(activation: str = 'relu') -> nn.Module:
    """Get the specified activation layer. 

    Args:
        activation (str): one of ``'relu'``, ``'leaky_relu'``, ``'elu'``, ``'gelu'``, 
            ``'swish'`` and ``'efficient_swish'``. Default: ``'relu'``
    """
    assert activation in ["relu", "leaky_relu", "elu", "gelu", 
                          "swish", "efficient_swish"]
    activation_dict = {
        "relu": nn.ReLU(inplace=True),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.1, inplace=True),
        "elu": nn.ELU(alpha=1.0, inplace=True),
        "gelu": nn.GELU(),
        "swish": Swish(),
        "efficient_swish": MemoryEfficientSwish(),
    }
    return activation_dict[activation]

#----------------------
# Normalization Layers
#----------------------
def get_norm(norm: str, out_channels: int, bn_momentum: float = 0.1) -> nn.Module:
    """Get the specified normalization layer.

    Args:
        norm (str): one of BN or GN;
        out_channels (int): channel number.
    Returns:
        nn.Module: the normalization layer
    """
    assert norm in ["BN", "SyncBN", "GN", "IN"]
    norm = {
        "BN": nn.BatchNorm3d,
        "SyncBN": nn.BatchNorm3d, 
        "IN": nn.InstanceNorm3d,
        "GN": lambda channels: nn.GroupNorm(16, channels),
        }[norm]
    if norm in ["BN", "IN"]:
        return norm(out_channels, momentum=bn_momentum)
    else:
        return norm(out_channels)
