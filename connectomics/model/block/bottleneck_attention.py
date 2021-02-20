import math
import torch
from torch import nn, einsum

from einops import rearrange

# modification of code from https://github.com/lucidrains/bottleneck-transformer-pytorch

# positional embedding helpers

def expand_dims(t, dims, ks):
    for d in dims:
        t = t.unsqueeze(dim = d)
    expand_shape = [-1] * len(t.shape)
    for d,k in zip(dims, ks):
        expand_shape[d] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, d, h, w, dim = q.shape
    logits = einsum('b h x y z d, r d -> b h x y z r', q, rel_k)
    logits = rearrange(logits, 'b h x y z r -> b (h x y) z r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, d, h, w, w)
    logits = expand_dims(logits, dims = [3, 5], ks = [d, h])
    return logits

# positional embeddings

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        depth, height, width = fmap_size
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_depth = nn.Parameter(torch.randn(depth * 2 - 1, dim_head) * scale)
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        d, h, w = self.fmap_size

        q = rearrange(q, 'b h (x y z) d -> b h x y z d', x = d, y = h, z = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j z k -> b h (x y z) (i j k)') # ??

        q = rearrange(q, 'b h x y z d -> b h x z y d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i z k y j -> b h (x z y) (i k j)') # ?
        
        q = rearrange(q, 'b h x z y d -> b h y z x d')
        rel_logits_d = relative_logits_1d(q, self.rel_depth)
        rel_logits_d = rearrange(rel_logits_d, 'b h y j z k x i -> b h (y z x) (j k i)')
        return rel_logits_w + rel_logits_h + rel_logits_d

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        depth, height, width = fmap_size
        scale = dim_head ** -0.5
        self.depth = nn.Parameter(torch.randn(depth, dim_head) * scale)
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        emb = rearrange(self.depth, 'dp d -> dp () () d') + rearrange(self.height, 'h d -> () h () d') + rearrange(self.width, 'w d -> () () w d')
        emb = rearrange(emb, ' dp h w d -> (dp h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

# classes

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Conv3d(dim, inner_dim * 3, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(fmap_size, dim_head)

    def forward(self, fmap):
        heads, b, c, d, h, w = self.heads, *fmap.shape

        q, k, v = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y z -> b h (x y z) d', h = heads), (q, k, v))

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', x = d, y = h, z = w)
        return out

class BottleBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out,
        proj_factor,
        downsample,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
        activation = nn.ReLU(),
        pad_mode = 'replicate'
    ):
        super().__init__()

        self.fmap_size = fmap_size
        
        # shortcut
        if dim != dim_out or downsample:
            kernel_size, stride, padding = (3, 2, 1) if downsample else (1, 1, 0)

            self.shortcut = nn.Sequential(
                nn.Conv3d(dim, dim_out, kernel_size, stride = stride, padding = padding, bias = False, padding_mode = pad_mode),
                nn.BatchNorm3d(dim_out),
                activation
            )
        else:
            self.shortcut = nn.Identity()

        # contraction and expansion

        attn_dim_in = dim_out // proj_factor
        attn_dim_out = heads * dim_head

        self.net = nn.Sequential(
            nn.Conv3d(dim, attn_dim_in, 1, bias = False),
            nn.BatchNorm3d(attn_dim_in),
            activation,
            Attention(
                dim = attn_dim_in,
                fmap_size = fmap_size,
                heads = heads,
                dim_head = dim_head,
                rel_pos_emb = rel_pos_emb
            ),
            nn.AvgPool3d(kernel_size, stride=stride, padding=padding) if downsample else nn.Identity(),
            nn.BatchNorm3d(attn_dim_out),
            activation,
            nn.Conv3d(attn_dim_out, dim_out, 1, bias = False),
            nn.BatchNorm3d(dim_out)
        )

        # init last batch norm gamma to zero

        nn.init.zeros_(self.net[-1].weight)

        # final activation

        self.activation = activation

    def forward(self, x):
        _, _, d, h, w = x.shape
        assert d == self.fmap_size[0] and h == self.fmap_size[1] and w == self.fmap_size[2], f'depth, height, and width [{d} {h} {w}] of feature map must match the fmap_size given at init {self.fmap_size}'
        shortcut = self.shortcut(x)
        x = self.net(x)
        x += shortcut
        return self.activation(x)

# main bottle stack

class BottleStack(nn.Module):
    def __init__(
        self,
        *,
        dim,
        fmap_size,
        dim_out = 2048,
        proj_factor = 4,
        num_layers = 3,
        heads = 4,
        dim_head = 128,
        downsample = True,
        rel_pos_emb = False,
        activation = nn.ReLU()
    ):
        super().__init__()

        self.dim = dim
        self.fmap_size = fmap_size

        layers = []

        for i in range(num_layers):
            is_first = i == 0
            dim = (dim if is_first else dim_out)
            layer_downsample = is_first and downsample

            fmap_divisor = (2 if downsample and not is_first else 1)
            layer_fmap_size = tuple(map(lambda t: t // fmap_divisor, fmap_size))

            layers.append(BottleBlock(
                dim = dim,
                fmap_size = layer_fmap_size,
                dim_out = dim_out,
                proj_factor = proj_factor,
                heads = heads,
                dim_head = dim_head,
                downsample = layer_downsample,
                rel_pos_emb = rel_pos_emb,
                activation = activation
            ))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        _, c, d, h, w = x.shape
        assert c == self.dim, f'channels of feature map {c} must match channels given at init {self.dim}'
        assert d == self.fmap_size[0] and h == self.fmap_size[1] and w == self.fmap_size[2], f'depth, height, and width [{d} {h} {w}] of feature map must match the fmap_size given at init {self.fmap_size}'
        return self.net(x)
