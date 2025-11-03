"""
Plug-and-Play Progressive Focused Attention (PFA) module extracted from pft_arch.

This file provides a self-contained attention block that can be dropped into
other models. It mirrors the orange/light-orange boxes in the PFT structure
figure: sparse attention with top-k selection, Hadamard reweighting by PFA maps,
and optional CUDA-accelerated sparse matrix multiplications (SMM).

If the CUDA extension `smm_cuda` is not available, the code falls back to a
pure PyTorch implementation using dense ops + gather/scatter so it can run
anywhere (slower but functionally equivalent for testing and small inputs).
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
import importlib.util

# Detect availability of the custom CUDA extension without importing it,
# to keep static analyzers quiet and avoid import-time errors.
_HAS_SMM = importlib.util.find_spec("smm_cuda") is not None


class _SMM_QmK_Fallback(Function):
    """Fallback SMM for Q @ K^T with sparse column indices (top-k per row).

    Inputs are shaped like the CUDA path expects after view/reshape:
      - A: (B*H, N, C)
      - B: (B*H, C, N)
      - index: (B*H, N, K)  indices along the last dim of N (columns)
    Returns (B*H, N, K)
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # Compute dense logits then gather K columns
        logits = torch.bmm(A, B)  # (B*H, N, N)
        out = torch.gather(logits, dim=-1, index=index.long())
        ctx.save_for_backward(A, B, index)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        A, B, index = ctx.saved_tensors
        # Scatter grad to dense (B*H, N, N)
        grad_logits = torch.zeros(A.size(0), A.size(1), B.size(2), device=A.device, dtype=A.dtype)
        grad_logits.scatter_(-1, index.long(), grad_output)
        # d(A) = d(logits) @ B^T;  d(B) = A^T @ d(logits)
        grad_A = torch.bmm(grad_logits, B.transpose(-2, -1))
        grad_B = torch.bmm(A.transpose(-2, -1), grad_logits)
        return grad_A, grad_B, None


class _SMM_AmV_Fallback(Function):
    """Fallback SMM for A @ V with sparse column indices.

    Inputs after view/reshape:
      - A: (B*H, N, K)
      - V: (B*H, N, C)
      - index: (B*H, N, K)  indices along last dim of N (columns of V)
    Returns (B*H, N, C)
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, V: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # Gather selected columns of V per row, then weighted sum by A
        gathered = torch.gather(V, dim=1, index=index.long().unsqueeze(-1).expand(-1, -1, -1, V.size(-1)))
        # gathered: (B*H, N, K, C)
        out = torch.einsum('bnk,bnkc->bnc', A, gathered)
        ctx.save_for_backward(A, V, index)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        A, V, index = ctx.saved_tensors
        # Compute grads by reconstructing gathered then scattering
        gathered = torch.gather(V, dim=1, index=index.long().unsqueeze(-1).expand(-1, -1, -1, V.size(-1)))
        # dA = grad_out dot gathered
        grad_A = torch.einsum('bnc,bnkc->bnk', grad_output, gathered)
        # d(gathered) = outer(A, grad_out)
        d_gathered = torch.einsum('bnk,bnc->bnkc', A, grad_output)
        # scatter back to V
        grad_V = torch.zeros_like(V)
        grad_V.scatter_add_(1, index.long().unsqueeze(-1).expand_as(d_gathered), d_gathered)
        return grad_A, grad_V, None


class SMM_QmK(Function):
    """CUDA SMM wrapper with fallback."""

    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        if _HAS_SMM:
            import smm_cuda  # type: ignore  # local import to avoid linter error when missing
            return smm_cuda.SMM_QmK_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())
        return _SMM_QmK_Fallback.apply(A, B, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        if _HAS_SMM:
            import smm_cuda  # type: ignore
            grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
                grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
            )
            return grad_A, grad_B, None
        return _SMM_QmK_Fallback.backward.__func__(ctx, grad_output)  # type: ignore


class SMM_AmV(Function):
    """CUDA SMM wrapper with fallback."""

    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        if _HAS_SMM:
            import smm_cuda  # type: ignore
            return smm_cuda.SMM_AmV_forward_cuda(A.contiguous(), B.contiguous(), index.contiguous())
        return _SMM_AmV_Fallback.apply(A, B, index)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        if _HAS_SMM:
            import smm_cuda  # type: ignore
            grad_A, grad_B = smm_cuda.SMM_AmV_backward_cuda(
                grad_output.contiguous(), A.contiguous(), B.contiguous(), index.contiguous()
            )
            return grad_A, grad_B, None
        return _SMM_AmV_Fallback.backward.__func__(ctx, grad_output)  # type: ignore


class ProgressiveFocusedAttention(nn.Module):
    """
    Plug-and-play PFA module (windowed self-attention with progressive focusing).

    Args:
        dim: channel dim C
        num_heads: number of heads H
        window_size: (Wh, Ww)
        layer_id: global layer id for selecting top-k per layer
        num_topk: list/tuple with per-layer k values
        qkv_bias: add bias to q/k/v linear in the caller before packing

    Forward signature mirrors the original implementation:
        forward(qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0)
        - qkvp: (B*nw, N, 4C) packed [Q,K,V,V_lepe]
        - pfa_values / pfa_indices: lists with two slots [no_shift, shift]
        - rpi: relative position index (Wh*Ww, Wh*Ww)
        - mask: attention mask for SW-MSA when shifted
        - shift: 0/1 selects which PFA slot to use
    """

    def __init__(self, dim: int, layer_id: int, window_size, num_heads: int, num_topk, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.num_topk = num_topk

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-20

        # relative position bias table
        if dim > 100:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            )
        else:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), 1)
            )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift: int = 0):
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        # Pack [Q, K, V, V_lepe] then split into heads. Shapes after view:
        #   q,k,v,v_lepe: (B*nw, Heads, N, C/Heads)
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale

        # Dense attention path when没有已有稀疏索引（第一层或未启用top-k）。
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            attn = attn + relative_position_bias
            # Apply SW-MSA 的窗口移位遮罩：在 shift=1 分支才生效。
            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            # 稀疏注意力路径：仅计算被上一层保留下来的 top-k 位置，显著降低复杂度。
            topk = pfa_indices[shift].shape[-1]
            qv = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            kv = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(qv, kv, smm_index).view(b_, self.num_heads, n, topk)

            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(
                b_, self.num_heads, n, n
            )
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            attn = attn + relative_position_bias

        attn = self.softmax(attn)

        # Progressive focusing：使用上一层保存的注意力值对当前注意力做Hadamard重加权并归一化。
        if pfa_values[shift] is not None:
            attn = (attn * pfa_values[shift])
            attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)

        # 本层的稀疏化：按行选取 top-k，并更新索引以供下一层复用。
        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn

        # 根据是否稀疏，选择 A@V 的计算路径：密集 matmul 或 CUDA 稀疏乘（带CPU回退）。
        if pfa_indices[shift] is None:
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            topk = pfa_indices[shift].shape[-1]
            attn_ = attn.view(b_ * self.num_heads, n, topk)
            v_ = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn_, v_, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe).transpose(1, 2).reshape(b_, n, c)

        x = self.proj(x)
        return x, pfa_values, pfa_indices


def _demo_main():
    """Minimal self-test with random tensors.

    Run with:
        conda activate torchv5
        python basicsr/archs/pfa_module.py
    """
    torch.manual_seed(0)
    window_size = 8
    dim = 64
    num_heads = 8
    layer_id = 0
    n = window_size * window_size
    b_times_nw = 2  # num_windows * batch

    # Per-layer top-k schedule（示例保持全量，便于快速验证数值正确性）
    num_topk = [n] * 24

    # Relative position index 计算（与 PFT 的 calculate_rpi_sa 等价）
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    rpi = relative_coords.sum(-1)

    # Fake q,k,v and local positional v (lepe)
    c = dim
    qkv = torch.randn(b_times_nw, n, 3 * c)
    v_lepe = torch.randn(b_times_nw, n, c)
    qkvp = torch.cat([qkv, v_lepe], dim=-1)

    pfa_values = [None, None]
    pfa_indices = [None, None]

    attn = ProgressiveFocusedAttention(
        dim=dim,
        layer_id=layer_id,
        window_size=window_size,
        num_heads=num_heads,
        num_topk=num_topk,
    )

    out, pfa_values, pfa_indices = attn(qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0)
    print("Output shape:", tuple(out.shape))
    print("PFA map shape:", tuple(pfa_values[0].shape) if pfa_values[0] is not None else None)
    print("Indices present:", pfa_indices[0] is not None)
    print("Using CUDA SMM:", _HAS_SMM)


if __name__ == "__main__":
    _demo_main()


