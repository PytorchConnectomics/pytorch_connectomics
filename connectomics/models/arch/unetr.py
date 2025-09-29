from __future__ import annotations

from collections.abc import Sequence
import numpy as np

import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep


class UNETR(nn.Module):
    """
    UNETR model, adapted from: Hatamizadeh et al., <https://arxiv.org/abs/2103.10504>.

    Args:
        in_channel (int): dimension of input channels.
        out_channel (int): dimension of output channels.
        img_size (Sequence[int] | int): dimension of input image.
        patch_size (Sequence[int] | int): dimension of patch size.
        feature_size (int): dimension of network feature size. Defaults to 16.
        hidden_size (int): dimension of hidden layer. Defaults to 768.
        mlp_dim (int): dimension of feedforward layer. Defaults to 3072.
        num_heads (int): number of attention heads. Defaults to 12.
        pos_embed (str): position embedding layer type. Defaults to ```'conv'```.
        norm_name (tuple | str): feature normalization type and arguments. Defaults to ```'instance'```.
        is_isotropic (bool): whether the whole model is isotropic. Default: False
        conv_block (bool): if convolutional block is used. Defaults to True.
        res_block (bool): if residual block is used. Defaults to True.
        dropout_rate (float): fraction of the input units to drop. Defaults to 0.0.
        spatial_dims (int): number of spatial dims. Defaults to 3.
        qkv_bias (bool): apply the bias term for the qkv linear layer in self attention block. Defaults to False.
        save_attn (bool): to make accessible the attention in self attention block. Defaults to False.
    """

    def __init__(
        self,
        in_channel: int = 1,
        out_channel: int = 3,
        img_size: Sequence[int] | int = (64, 128, 128),
        patch_size: Sequence[int] | int = (16, 16, 16),
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = 'conv',
        norm_name: tuple | str = 'instance',
        is_isotropic: bool = False,
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
        **kwargs,
    ) -> None:

        super().__init__()

        assert 0 <= dropout_rate <= 1
        assert hidden_size % num_heads == 0
        assert pos_embed in ['conv', 'perceptron']
        assert np.prod(np.asarray(patch_size) % 4) == 0

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        upsample_kernel_size = 2
        if not is_isotropic:
            patch_size = (patch_size[0] // 4, patch_size[1], patch_size[2])
            upsample_kernel_size = (1, 2, 2)
        self.patch_size = patch_size
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        self.vit = ViT(
            in_channels=in_channel,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            pos_embed=pos_embed,
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channel,
            out_channels=feature_size,
            kernel_size=(1, 3, 3),  # exclusively use 2D conv at finest scale
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        encoder2_1 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        encoder2_2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size * 2,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder2 = nn.Sequential(encoder2_1, encoder2_2)
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=(1, 3, 3),  # exclusively use 2D conv at finest scale
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims,
                                in_channels=feature_size,
                                out_channels=out_channel)  # type: ignore
        self.proj_axes = (0, spatial_dims + 1) + tuple(
            d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)
