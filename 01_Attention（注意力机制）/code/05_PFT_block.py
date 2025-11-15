'''
Standalone plug-and-play test for PFT building blocks.
Copy of original implementations from basicsr/archs/pft_arch.py
with an optional CUDA fallback and a simple main() demo.
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import importlib
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
import collections.abc
from itertools import repeat
import importlib.util

# from basicsr.archs.arch_util import to_2tuple, trunc_normal_


# 工具函数：将输入转换为 2 元组
def to_2tuple(x):
    """将输入转换为 2 元组。

    如果输入是可迭代对象，直接转换为元组；
    否则，将输入重复 2 次形成元组。
    """
    if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
        return tuple(x)
    return tuple(repeat(x, 2))


# 工具函数：截断正态分布初始化
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """截断正态分布初始化（无梯度版本）。"""

    def norm_cdf(x):
        # 计算标准正态分布的累积分布函数
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # 使用截断均匀分布生成值，然后使用逆 CDF 转换为正态分布
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # 用 [low, up] 中的值均匀填充张量，然后转换为 [2l-1, 2u-1]
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # 使用逆 CDF 变换得到截断标准正态分布
        tensor.erfinv_()

        # 转换为适当的均值、标准差
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # 截断以确保在适当范围内
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """用截断正态分布的值填充输入张量。

    Args:
        tensor: n 维 torch.Tensor
        mean: 正态分布的均值
        std: 正态分布的标准差
        a: 最小截断值
        b: 最大截断值
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# ------------------------- Optional CUDA fallback ------------------------- #
_spec = importlib.util.find_spec('smm_cuda')
if _spec is not None:
    smm_cuda = importlib.import_module('smm_cuda')  # type: ignore
    _HAS_SMM_CUDA = True
else:
    smm_cuda = None
    _HAS_SMM_CUDA = False


# ------------------------- Original code (copied) ------------------------- #
class SMM_QmK(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        ctx.has_cuda_impl = _HAS_SMM_CUDA
        if _HAS_SMM_CUDA:
            return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

        # Fallback: dense attention then gather
        full_attn = torch.bmm(A.contiguous(), B.contiguous())  # (bh, n, n)
        index_long = index.long()
        return torch.gather(full_attn, dim=-1, index=index_long)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        if getattr(ctx, 'has_cuda_impl', False) and smm_cuda is not None:
            grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
                grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
            )
            return grad_A, grad_B, None

        # Fallback: gradients not supported (sufficient for inference demo)
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        return grad_A, grad_B, None


class SMM_AmV(Function):
    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        ctx.has_cuda_impl = _HAS_SMM_CUDA
        if _HAS_SMM_CUDA:
            return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())

        # Fallback: manual sparse gather
        bhn, n, topk = A.shape
        d = B.shape[-1]
        index_long = index.long()
        output = torch.zeros(bhn, n, d, device=A.device, dtype=A.dtype)
        for b in range(bhn):
            for i in range(n):
                cols = index_long[b, i]
                gathered = B[b, cols]
                output[b, i] = (A[b, i].unsqueeze(0) @ gathered).squeeze(0)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        if getattr(ctx, 'has_cuda_impl', False) and smm_cuda is not None:
            grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
                grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
            )
            return grad_A, grad_B, None

        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(B)
        return grad_A, grad_B, None


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, layer_id, window_size, num_heads, num_topk, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = window_size
        self.num_heads = num_heads
        self.num_topk = num_topk
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        if dim > 100:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))
        else:
            self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), 1))
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp

        q = q * self.scale
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            topk = pfa_indices[shift].shape[-1]
            q = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            k = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(q, k, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(b_, self.num_heads, n, n)
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            if not self.training:
                attn.add_(relative_position_bias)
            else:
                attn = attn + relative_position_bias

        if not self.training:
            attn = torch.softmax(attn, dim=-1, out=attn)
        else:
            attn = self.softmax(attn)

        if pfa_values[shift] is not None:
            if not self.training:
                attn.mul_(pfa_values[shift])
                attn.add_(self.eps)
                denom = attn.sum(dim=-1, keepdim=True).add_(self.eps)
                attn.div_(denom)
            else:
                attn = (attn * pfa_values[shift])
                attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn

        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn = attn.view(b_ * self.num_heads, n, topk)
            v = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn, v, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe)
            x = x.transpose(1, 2).reshape(b_, n, c)

        if not self.training:
            del q, k, v, relative_position_bias
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        x = self.proj(x)
        return x, pfa_values, pfa_indices


class PFTransformerLayer(nn.Module):
    def __init__(self,
                 dim,
                 block_id,
                 layer_id,
                 input_resolution,
                 num_heads,
                 num_topk,
                 window_size,
                 shift_size,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.convlepe_kernel_size = convffn_kernel_size
        self.v_LePE = dwconv(hidden_features=dim, kernel_size=self.convlepe_kernel_size)

        self.attn_win = WindowAttention(
            self.dim,
            layer_id=layer_id,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            num_topk=num_topk,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer)

    def forward(self, x, pfa_list, x_size, params):
        pfa_values, pfa_indices = pfa_list[0], pfa_list[1]
        h, w = x_size
        b, n, c = x.shape
        c4 = 4 * c

        shortcut = x

        x = self.norm1(x)
        x_qkv = self.wqkv(x)

        v_lepe = self.v_LePE(torch.split(x_qkv, c, dim=-1)[-1], x_size)
        x_qkvp = torch.cat([x_qkv, v_lepe], dim=-1)

        if self.shift_size > 0:
            shift = 1
            shifted_x = torch.roll(x_qkvp.reshape(b, h, w, c4), shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shift = 0
            shifted_x = x_qkvp.reshape(b, h, w, c4)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c4)
        attn_windows, pfa_values, pfa_indices = self.attn_win(
            x_windows,
            pfa_values=pfa_values,
            pfa_indices=pfa_indices,
            rpi=params['rpi_sa'],
            mask=params['attn_mask'],
            shift=shift
        )
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        x_win = attn_x

        x = shortcut + x_win.view(b, n, c)
        x = x + self.convffn(self.norm2(x), x_size)

        pfa_list = [pfa_values, pfa_indices]
        return x, pfa_list


# ------------------------- Helper functions ------------------------- #
def calculate_rpi_sa(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    relative_position_index = relative_coords.sum(-1)
    return relative_position_index


def calculate_mask(x_size, window_size):
    h, w = x_size
    img_mask = torch.zeros((1, h, w, 1))
    h_slices = (slice(0, -window_size), slice(-window_size, -(window_size // 2)), slice(-(window_size // 2), None))
    w_slices = (slice(0, -window_size), slice(-window_size, -(window_size // 2)), slice(-(window_size // 2), None))
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


# ------------------------- Minimal demo ------------------------- #
if __name__ == '__main__':
    torch.manual_seed(42)

    dim = 96
    layer = PFTransformerLayer(
        dim=dim,
        block_id=0,
        layer_id=0,
        input_resolution=(224, 224),
        num_heads=6,
        num_topk=[256] * 24,
        window_size=8,
        shift_size=0,
        convffn_kernel_size=5,
        mlp_ratio=2.0,
        qkv_bias=True,
    )

    dummy = torch.randn(1, 3, 224, 224)
    conv_first = nn.Conv2d(3, dim, 3, 1, 1)
    conv_last = nn.Conv2d(dim, 3, 3, 1, 1)

    feat = conv_first(dummy)
    tokens = feat.flatten(2).transpose(1, 2)

    pfa_values = [None, None]
    pfa_indices = [None, None]
    params = {
        'attn_mask': calculate_mask((224, 224), 8),
        'rpi_sa': calculate_rpi_sa(8)
    }

    out_tokens, _ = layer(tokens, [pfa_values, pfa_indices], (224, 224), params)

    out_feat = out_tokens.transpose(1, 2).view(1, dim, 224, 224)
    out_image = conv_last(out_feat)

    print('Input image:', dummy.shape)
    print('Output image:', out_image.shape)
    print('Test passed:', out_image.shape == dummy.shape)
