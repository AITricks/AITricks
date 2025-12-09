import torch
import torch.nn as nn
import timm
from torch import Tensor

from .irpe import build_rpe, get_rpe_config


class RPEAttention(nn.Module):
    """Self-attention enhanced with image relative position encoding (PAT_sf)."""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.0, rpe_config=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rpe_q, self.rpe_k, self.rpe_v = build_rpe(
            rpe_config, head_dim=head_dim, num_heads=num_heads
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)
        _, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = q @ k.transpose(-2, -1)

        if self.rpe_k is not None:
            attn += self.rpe_k(q, h, w)
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale).transpose(2, 3)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        if self.rpe_v is not None:
            out += self.rpe_v(attn)

        x = out.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2).view(b, c, h, w)
        return x


class SRM(nn.Module):
    """Style pooling based channel attention module used in PAT_ch."""

    def __init__(self, channel):
        super().__init__()
        self.cfc1 = nn.Conv2d(channel, channel, kernel_size=(1, 2), bias=False)
        self.bn = nn.BatchNorm2d(channel)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        mean = x.reshape(b, c, -1).mean(-1).view(b, c, 1, 1)
        std = x.reshape(b, c, -1).std(-1).view(b, c, 1, 1)
        u = torch.cat([mean, std], dim=-1)
        z = self.cfc1(u)
        z = self.bn(z)
        g = self.sigmoid(z).reshape(b, c, 1, 1)
        return x * g


class Partial_conv3(nn.Module):
    """Channel mixing block; becomes PAT_ch or PAT_sf depending on channel_type."""

    def __init__(self, dim, n_div, forward_type, use_attn='', channel_type='',
                 patnet_t0=False):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim = dim
        self.n_div = n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.use_attn = use_attn
        self.channel_type = channel_type

        if use_attn:
            if channel_type == 'self':
                rpe_config = get_rpe_config(
                    ratio=20,
                    method="euc",
                    mode='bias',
                    shared_head=False,
                    skip=0,
                    rpe_on='k',
                )
                num_heads = 4 if patnet_t0 else 6
                self.attn = RPEAttention(
                    self.dim_untouched,
                    num_heads=num_heads,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    rpe_config=rpe_config,
                )
                self.norm = timm.layers.LayerNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
            elif channel_type == 'se':
                self.attn = SRM(self.dim_untouched)
                self.norm = nn.BatchNorm2d(self.dim_untouched)
                self.forward = self.forward_atten
        else:
            if forward_type == 'slicing':
                self.forward = self.forward_slicing
            elif forward_type == 'split_cat':
                self.forward = self.forward_split_cat
            else:
                raise NotImplementedError

    def forward_atten(self, x: Tensor) -> Tensor:
        if self.channel_type:
            x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
            x1 = self.partial_conv3(x1)
            x2 = self.norm(x2)
            x2 = self.attn(x2)
            x = torch.cat((x1, x2), 1)
        return x

    def forward_slicing(self, x: Tensor) -> Tensor:
        x1 = x.clone()
        x1[:, :self.dim_conv3, :, :] = self.partial_conv3(x1[:, :self.dim_conv3, :, :])
        return x1

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class partial_spatial_attn_layer_reverse(nn.Module):
    """Spatial attention refinement corresponding to PAT_sp."""

    def __init__(self, dim, n_head, partial=0.5):
        super().__init__()
        self.dim = dim
        self.dim_conv = int(partial * dim)
        self.dim_untouched = dim - self.dim_conv
        self.nhead = n_head
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, 1, bias=False)
        self.conv_attn = nn.Conv2d(self.dim_untouched, n_head, 1, bias=False)
        self.norm = nn.BatchNorm2d(self.dim_untouched)
        self.norm2 = nn.BatchNorm2d(self.dim_conv)
        self.act = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_untouched, self.dim_conv], 1)
        weight = self.act(self.conv_attn(x1))
        x1 = self.norm(x1 * weight)
        x2 = self.conv(self.norm2(x2))
        x = torch.cat((x1, x2), 1)
        return x


def main():
    torch.manual_seed(0)
    dummy = torch.randn(1, 64, 32, 32)

    pat_ch = Partial_conv3(
        dim=64, n_div=4, forward_type='split_cat',
        use_attn=True, channel_type='se'
    )
    pat_sf = Partial_conv3(
        dim=64, n_div=4, forward_type='split_cat',
        use_attn=True, channel_type='self', patnet_t0=False
    )
    pat_sp = partial_spatial_attn_layer_reverse(dim=64, n_head=1, partial=0.5)

    pat_ch.eval()
    pat_sf.eval()
    pat_sp.eval()

    with torch.no_grad():
        out_ch = pat_ch(dummy)
        out_sf = pat_sf(dummy)
        out_sp = pat_sp(dummy)

    print('PAT_ch output shape:', out_ch.shape)
    print('PAT_sf output shape:', out_sf.shape)
    print('PAT_sp output shape:', out_sp.shape)


if __name__ == "__main__":
    main()

