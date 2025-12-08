"""
Reusable attention modules extracted from the MFF-Net implementation.

The modules correspond to the plug-and-play components highlighted in the
paper diagrams:
    - CrossAttention: cross-branch attention inside the MSFU block (Fig. 3).
    - DirectionalConvUnit: directional convolution stack used before SW-SAM.
    - SWSAM: sliding window spatial attention used in the encoder fusion (Fig. 2).
Each module can be imported independently in other projects.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossAttention(nn.Module):
    """
    Cross attention unit from the decoder. It treats one feature map as query
    tokens and another as key/value tokens, enabling cross-scale fusion.
    """

    def __init__(
        self,
        dim1: int,
        dim2: int,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.scale = qk_scale or head_dim ** -1.0

        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim1)

        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, fea: Tensor, aux_fea: Tensor) -> Tensor:
        """
        Args:
            fea: Tensor of shape [B, N1, dim1] used to build queries.
            aux_fea: Tensor of shape [B, N2, dim2] used to build keys/values.
        """

        _, n_tokens, _ = fea.shape
        batch_size, aux_tokens, _ = aux_fea.shape
        channels = self.dim

        q1 = (
            self.q1(fea)
            .reshape(batch_size, n_tokens, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k2 = (
            self.k2(aux_fea)
            .reshape(batch_size, aux_tokens, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v2 = (
            self.v2(aux_fea)
            .reshape(batch_size, aux_tokens, self.num_heads, channels // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        fea = (attn @ v2).transpose(1, 2).reshape(batch_size, n_tokens, channels)
        fea = self.proj(fea)
        fea = self.proj_drop(fea)
        return fea


class DirectionalConvUnit(nn.Module):
    """
    Directional convolution block that applies convolutions along horizontal,
    vertical, and both diagonal directions before concatenating the responses.
    It expands contextual information while keeping the number of channels.
    """

    def __init__(self, channel: int) -> None:
        super().__init__()

        self.h_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        return torch.cat((x1, x2, x3, x4), dim=1)

    # The following helpers are adapted from CoANet with light modifications.
    def h_transform(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2] + shape[3] - 1)
        return x

    def inv_h_transform(self, x: Tensor) -> Tensor:
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3] + 1)
        return x[..., 0 : shape[3] - shape[2] + 1]

    def v_transform(self, x: Tensor) -> Tensor:
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2] + shape[3] - 1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x: Tensor) -> Tensor:
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = F.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3] + 1)
        x = x[..., 0 : shape[3] - shape[2] + 1]
        return x.permute(0, 1, 3, 2)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batch_size, -1, height, width)


class SWSAM(nn.Module):
    """
    Sliding Window Spatial Attention Module. It splits channels into four
    groups, applies independent spatial attention, and fuses responses with
    learnable weights plus a refinement conv.
    """

    def __init__(self, channel: int = 32) -> None:
        super().__init__()
        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(
            BasicConv2d(1, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = channel_shuffle(x, 4)
        _, channels, _, _ = x.shape
        split_size = int(channels / 4)
        x1, x2, x3, x4 = torch.split(x, split_size, dim=1)
        s1 = self.SA1(x1)
        s2 = self.SA2(x2)
        s3 = self.SA3(x3)
        s4 = self.SA4(x4)
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1] + s3 * nor_weights[2] + s4 * nor_weights[3]
        return self.sa_fusion(s_all) * x + x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test SWSAM
    swsam = SWSAM(channel=64).to(device)
    swsam_input = torch.randn(1, 64, 64, 64, device=device)
    swsam_output = swsam(swsam_input)
    print("SWSAM output:", swsam_output.shape)

    # Test DirectionalConvUnit
    dcu = DirectionalConvUnit(channel=64).to(device)
    dcu_output = dcu(swsam_input)
    print("DirectionalConvUnit output:", dcu_output.shape)

    # Test CrossAttention
    cross_attn = CrossAttention(dim1=128, dim2=256, dim=128, num_heads=4).to(device)
    query_feats = torch.randn(1, 64, 128, device=device)
    kv_feats = torch.randn(1, 64, 256, device=device)
    attn_output = cross_attn(query_feats, kv_feats)
    print("CrossAttention output:", attn_output.shape)

