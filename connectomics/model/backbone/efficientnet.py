import torch
import torch.nn as nn
# from ..block.pooling_attention import *
from ..block.partialconv3d import PartialConv3d
from ..block.depthwise import InvertedResidual, InvertedResidualDilated, dw_stack
from ..utils import get_activation, get_norm_3d, get_conv


class EfficientNet(nn.Module):
    """EfficientNet backbone for 3D instance segmentation.
    """
    dilation_factors = [1, 2, 4, 8]
    num_stages = 5

    def __init__(self, block, blocks=[1, 2, 2, 2, 4], ks=[3, 3, 5, 5, 3], isotropy=[False, False, False, True, True], 
                 expansion_factor = 1, norm_mode='bn', filters=[32, 64, 96, 128, 160], 
                 in_channel=1, attention=None, bn_momentum=0.01, conv_type='standard', 
                 pad_mode='replicate', act_mode='elu', **kwargs):
        super(EfficientNet, self).__init__()
        self.inplanes = filters[0]

        if isinstance(block, InvertedResidualDilated):
            self.all_dilated = True
            self.conv1 = nn.ModuleList([get_conv(conv_type)(in_channel, self.inplanes // 4, kernel_size=3, 
                bias=False, stride=1, dilation=self.dilation_factors[i], padding=self.dilation_factors[i], 
                padding_mode=pad_mode) for i in range(4)])
        else:
            self.all_dilated = False
            self.conv1 = get_conv(conv_type)(in_channel, self.inplanes, kernel_size=3, stride=1, 
                                padding=1, padding_mode=pad_mode, bias=False)

        self.bn1 = get_norm_3d(norm_mode, self.inplanes, bn_momentum)
        self.relu = get_activation(act_mode)

        shared_args = {
            'expansion_factor': expansion_factor,
            'bn_momentum': bn_momentum,
            'norm_layer': norm_mode,
            'attention': attention,
            'conv_type': conv_type,
            'padding_mode': pad_mode,
            'activation': act_mode,
        }

        self.layer0 = dw_stack(block, filters[0], filters[0], kernel_size=ks[0], stride=1, 
                             repeats=blocks[0], isotropic=isotropy[0], shared_args=shared_args)
        self.layer1 = dw_stack(block, filters[0], filters[1], kernel_size=ks[1], stride=2, 
                             repeats=blocks[1], isotropic=isotropy[1], shared_args=shared_args)
        self.layer2 = dw_stack(block, filters[1], filters[2], kernel_size=ks[2], stride=2, 
                             repeats=blocks[2], isotropic=isotropy[2], shared_args=shared_args)
        self.layer3 = dw_stack(block, filters[2], filters[3], kernel_size=ks[3], stride=(1,2,2), 
                             repeats=blocks[3], isotropic=isotropy[3], shared_args=shared_args)
        self.layer4 = dw_stack(block, filters[3], filters[4], kernel_size=ks[4], stride=2, 
                             repeats=blocks[4], isotropic=isotropy[4], shared_args=shared_args)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        if self.all_dilated:
            x = self._conv_and_cat(x, self.conv1)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def _conv_and_cat(self, x, conv_layers):
        y = [conv(x) for conv in conv_layers]
        return torch.cat(y, dim=1)

    def forward(self, x):
        return self._forward_impl(x)


def _efficientnet(block, **kwargs):
    model = EfficientNet(block, **kwargs)
    num_param = sum([param.nelement() for param in model.parameters()])
    print('Total number of parameters: ', num_param)
    
    return model


def efficientnet(block_type='inverted_res', **kwargs):
    assert block_type in ['inverted_res', 'inverted_res_dilated']
    block_dict = {
        'inverted_res': InvertedResidual,
        'inverted_res_dilated': InvertedResidualDilated,
    }
    block = block_dict[block_type]
    return _efficientnet(block, **kwargs)