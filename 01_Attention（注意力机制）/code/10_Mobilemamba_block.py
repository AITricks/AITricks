import importlib.util
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

import pywt
from timm.models.vision_transformer import trunc_normal_
from timm.layers import DropPath

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
LIB_MAMBA_DIR = os.path.join(MODEL_DIR, "lib_mamba")

if "model" not in sys.modules:
    model_pkg = types.ModuleType("model")
    model_pkg.__path__ = [MODEL_DIR]
    sys.modules["model"] = model_pkg
if "model.lib_mamba" not in sys.modules:
    lib_mamba_pkg = types.ModuleType("model.lib_mamba")
    lib_mamba_pkg.__path__ = [LIB_MAMBA_DIR]
    sys.modules["model.lib_mamba"] = lib_mamba_pkg

spec = importlib.util.spec_from_file_location(
    "model.lib_mamba.vmambanew", os.path.join(LIB_MAMBA_DIR, "vmambanew.py")
)
vmambanew = importlib.util.module_from_spec(spec)
sys.modules["model.lib_mamba.vmambanew"] = vmambanew
spec.loader.exec_module(vmambanew)

SS2D = vmambanew.SS2D


def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters


def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super().__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class MBWTConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        stride=1,
        bias=True,
        wt_levels=1,
        wt_type="db1",
        ssm_ratio=1,
        forward_type="v05",
    ):
        super().__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(
            wt_type, in_channels, in_channels, torch.float
        )
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(
            inverse_wavelet_transform, filters=self.iwt_filter
        )

        self.global_atten = SS2D(
            d_model=in_channels,
            d_state=1,
            ssm_ratio=ssm_ratio,
            initialize="v2",
            forward_type=forward_type,
            channel_first=True,
            k_group=2,
        )
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels * 4,
                    in_channels * 4,
                    kernel_size,
                    padding="same",
                    stride=1,
                    dilation=1,
                    groups=in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )

        self.wavelet_scale = nn.ModuleList(
            [
                _ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1)
                for _ in range(self.wt_levels)
            ]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(
                torch.ones(in_channels, 1, 1, 1), requires_grad=False
            )
            self.do_stride = lambda x_in: F.conv2d(
                x_in,
                self.stride_filter,
                bias=None,
                stride=self.stride,
                groups=in_channels,
            )
        else:
            self.do_stride = None

    def forward(self, x):
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(
                shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4]
            )
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.global_atten(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class DWConv2d_BN_ReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, bn_weight_init=1):
        super().__init__()
        self.add_module(
            "dwconv3x3",
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
        )
        self.add_module("bn1", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "dwconv1x1",
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=in_channels,
                bias=False,
            ),
        )
        self.add_module("bn2", nn.BatchNorm2d(out_channels))

        nn.init.constant_(self.bn1.weight, bn_weight_init)
        nn.init.constant_(self.bn1.bias, 0)
        nn.init.constant_(self.bn2.weight, bn_weight_init)
        nn.init.constant_(self.bn2.bias, 0)


class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self,
        a,
        b,
        ks=1,
        stride=1,
        pad=0,
        dilation=1,
        groups=1,
        bn_weight_init=1,
    ):
        super().__init__()
        self.add_module("c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module("bn", torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)


def nearest_multiple_of_16(n):
    if n % 16 == 0:
        return n
    lower_multiple = (n // 16) * 16
    upper_multiple = lower_multiple + 16

    if (n - lower_multiple) < (upper_multiple - n):
        return lower_multiple
    else:
        return upper_multiple


class MobileMambaModule(torch.nn.Module):
    """
    Standalone MobileMamba interaction block that can be plugged into CNN stacks.

    Args:
        dim (int): channel dimension of the input feature map.
        global_ratio (float): ratio of channels allocated to global (Mamba) branch.
        local_ratio (float): ratio of channels allocated to local depth-wise conv branch.
        kernels (int): spatial kernel size for the local branch.
        ssm_ratio (int): expansion ratio for SS2D state space model.
        forward_type (str): SS2D forward kernel type.
    """

    def __init__(
        self,
        dim,
        global_ratio=0.25,
        local_ratio=0.25,
        kernels=3,
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()
        self.dim = dim
        self.global_channels = nearest_multiple_of_16(int(global_ratio * dim))
        if self.global_channels + int(local_ratio * dim) > dim:
            self.local_channels = dim - self.global_channels
        else:
            self.local_channels = int(local_ratio * dim)
        self.identity_channels = self.dim - self.global_channels - self.local_channels
        if self.local_channels != 0:
            self.local_op = DWConv2d_BN_ReLU(
                self.local_channels, self.local_channels, kernels
            )
        else:
            self.local_op = nn.Identity()
        if self.global_channels != 0:
            self.global_op = MBWTConv2d(
                self.global_channels,
                self.global_channels,
                kernels,
                wt_levels=1,
                ssm_ratio=ssm_ratio,
                forward_type=forward_type,
            )
        else:
            self.global_op = nn.Identity()

        self.proj = torch.nn.Sequential(
            torch.nn.ReLU(),
            Conv2d_BN(dim, dim, bn_weight_init=0),
        )

    def forward(self, x):
        x1, x2, x3 = torch.split(
            x, [self.global_channels, self.local_channels, self.identity_channels], dim=1
        )
        x1 = self.global_op(x1)
        x2 = self.local_op(x2)
        x = self.proj(torch.cat([x1, x2, x3], dim=1))
        return x


class MobileMambaBlock(torch.nn.Module):
    """
    Composite block with residual connections to demonstrate drop-in usage.
    """

    def __init__(
        self,
        ed,
        global_ratio=0.25,
        local_ratio=0.25,
        kernels=5,
        drop_path=0.0,
        ssm_ratio=1,
        forward_type="v052d",
    ):
        super().__init__()

        self.dw0 = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0)
        self.ffn0 = Conv2d_BN(ed, ed, bn_weight_init=0.0)

        self.mixer = MobileMambaModule(
            ed,
            global_ratio=global_ratio,
            local_ratio=local_ratio,
            kernels=kernels,
            ssm_ratio=ssm_ratio,
            forward_type=forward_type,
        )

        self.dw1 = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0.0)
        self.ffn1 = Conv2d_BN(ed, ed, bn_weight_init=0.0)

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.ffn0(self.dw0(x))
        x = self.mixer(x)
        x = self.ffn1(self.dw1(x))
        x = shortcut + self.drop_path(x)
        return x


def test_mobilemamba_module():
    module = MobileMambaModule(
        dim=192,
        global_ratio=0.8,
        local_ratio=0.2,
        kernels=7,
        ssm_ratio=2,
        forward_type="v052d",
    )
    dummy = torch.randn(2, 192, 14, 14)
    with torch.no_grad():
        out = module(dummy)
    print("Input shape:", dummy.shape)
    print("Output shape:", out.shape)


if __name__ == "__main__":
    test_mobilemamba_module()

