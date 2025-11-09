"""
即插即用渐进式聚焦注意力（Progressive Focused Attention, PFA）模块。

本模块从 PFT 架构中提取，提供了一个独立的注意力块，可以方便地集成到其他模型中。
实现了 PFT 结构图中的核心功能：
- 稀疏注意力计算（top-k 选择）
- 基于 PFA 映射的 Hadamard 重加权
- 可选的 CUDA 加速稀疏矩阵乘法（SMM）

如果 CUDA 扩展 `smm_cuda` 不可用，代码会自动回退到纯 PyTorch 实现
（使用密集操作 + gather/scatter），确保可以在任何环境下运行（速度较慢但功能等价）。
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import math
import collections.abc
from itertools import repeat
import importlib.util

# 检测自定义 CUDA 扩展的可用性，避免导入时的错误
_HAS_SMM = importlib.util.find_spec("smm_cuda") is not None


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


class _SMM_QmK_Fallback(Function):
    """Q @ K^T 的稀疏矩阵乘法回退实现（每行选择 top-k 列）。

    输入张量形状（经过 view/reshape 后）：
      - A: (B*H, N, C)  查询矩阵 Q
      - B: (B*H, C, N)  键矩阵 K 的转置
      - index: (B*H, N, K)  每行选择的 K 个列索引
    返回: (B*H, N, K)
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # 计算密集注意力分数，然后通过 gather 选择 top-k 列
        logits = torch.bmm(A, B)  # (B*H, N, N)
        out = torch.gather(logits, dim=-1, index=index.long())
        ctx.save_for_backward(A, B, index)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        A, B, index = ctx.saved_tensors
        # 将梯度散射回密集矩阵 (B*H, N, N)
        grad_logits = torch.zeros(A.size(0), A.size(1), B.size(2), device=A.device, dtype=A.dtype)
        grad_logits.scatter_(-1, index.long(), grad_output)
        # 计算梯度：d(A) = d(logits) @ B^T;  d(B) = A^T @ d(logits)
        grad_A = torch.bmm(grad_logits, B.transpose(-2, -1))
        grad_B = torch.bmm(A.transpose(-2, -1), grad_logits)
        return grad_A, grad_B, None


class _SMM_AmV_Fallback(Function):
    """A @ V 的稀疏矩阵乘法回退实现（使用稀疏列索引）。

    输入张量形状（经过 view/reshape 后）：
      - A: (B*H, N, K)  注意力权重矩阵（稀疏）
      - V: (B*H, N, C)  值矩阵
      - index: (B*H, N, K)  V 矩阵中每行选择的 K 个列索引
    返回: (B*H, N, C)
    """

    @staticmethod
    def forward(ctx, A: torch.Tensor, V: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        # 根据索引收集 V 的对应列，然后与注意力权重 A 做加权求和
        gathered = torch.gather(V, dim=1, index=index.long().unsqueeze(-1).expand(-1, -1, -1, V.size(-1)))
        # gathered: (B*H, N, K, C)
        out = torch.einsum('bnk,bnkc->bnc', A, gathered)
        ctx.save_for_backward(A, V, index)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        A, V, index = ctx.saved_tensors
        # 通过重构 gathered 然后散射回原矩阵来计算梯度
        gathered = torch.gather(V, dim=1, index=index.long().unsqueeze(-1).expand(-1, -1, -1, V.size(-1)))
        # dA = grad_out 与 gathered 的点积
        grad_A = torch.einsum('bnc,bnkc->bnk', grad_output, gathered)
        # d(gathered) = A 与 grad_out 的外积
        d_gathered = torch.einsum('bnk,bnc->bnkc', A, grad_output)
        # 散射回 V 矩阵
        grad_V = torch.zeros_like(V)
        grad_V.scatter_add_(1, index.long().unsqueeze(-1).expand_as(d_gathered), d_gathered)
        return grad_A, grad_V, None


class SMM_QmK(Function):
    """Q @ K^T 稀疏矩阵乘法的 CUDA 加速包装器，带 CPU 回退。"""

    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        if _HAS_SMM:
            import smm_cuda  # type: ignore  # 局部导入避免缺失时的 linter 错误
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
    """A @ V 稀疏矩阵乘法的 CUDA 加速包装器，带 CPU 回退。"""

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
    即插即用渐进式聚焦注意力模块（带窗口的自注意力，支持渐进式聚焦）。

    参数:
        dim: 通道维度 C
        num_heads: 注意力头数 H
        window_size: 窗口大小 (Wh, Ww)
        layer_id: 全局层 ID，用于选择该层的 top-k 值
        num_topk: 每层的 k 值列表/元组
        qkv_bias: 是否在打包前为 q/k/v 线性层添加偏置

    前向传播签名（与原始实现一致）:
        forward(qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0)
        - qkvp: (B*nw, N, 4C) 打包的 [Q,K,V,V_lepe]
        - pfa_values / pfa_indices: 包含两个槽位的列表 [无移位, 移位]
        - rpi: 相对位置索引 (Wh*Ww, Wh*Ww)
        - mask: SW-MSA 移位时的注意力遮罩
        - shift: 0/1 选择使用哪个 PFA 槽位
    """

    def __init__(self, dim: int, layer_id: int, window_size, num_heads: int, num_topk, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.layer_id = layer_id
        self.window_size = to_2tuple(window_size)
        self.num_heads = num_heads
        self.num_topk = num_topk

        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # 缩放因子，用于稳定注意力计算
        self.eps = 1e-20  # 防止除零的小常数

        # 相对位置偏置表（根据通道数选择不同的头数）
        if dim > 100:
            # 经典 SR 模型：每个头有独立的相对位置偏置
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
            )
        else:
            # 轻量级 SR 模型：所有头共享相对位置偏置
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), 1)
            )
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.proj = nn.Linear(dim, dim)  # 输出投影层
        self.softmax = nn.Softmax(dim=-1)
        self.topk = self.num_topk[self.layer_id]  # 当前层的 top-k 值

    def forward(self, qkvp, pfa_values, pfa_indices, rpi, mask=None, shift: int = 0):
        """
        前向传播。

        Args:
            qkvp: 打包的查询、键、值和局部位置编码，形状 (B*nw, N, 4C)
            pfa_values: 渐进式聚焦注意力值列表 [无移位, 移位]
            pfa_indices: 渐进式聚焦注意力索引列表 [无移位, 移位]
            rpi: 相对位置索引
            mask: SW-MSA 移位时的注意力遮罩
            shift: 0=无移位, 1=移位

        Returns:
            x: 输出特征，形状 (B*nw, N, C)
            pfa_values: 更新后的 PFA 值
            pfa_indices: 更新后的 PFA 索引
        """
        b_, n, c4 = qkvp.shape
        c = c4 // 4
        # 将打包的 [Q, K, V, V_lepe] 拆分并重塑为多头形式
        # 形状: q,k,v,v_lepe: (B*nw, Heads, N, C/Heads)
        qkvp = qkvp.reshape(b_, n, 4, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v, v_lepe = qkvp[0], qkvp[1], qkvp[2], qkvp[3]

        q = q * self.scale  # 缩放查询向量

        # 密集注意力路径：当没有已有稀疏索引时（第一层或未启用 top-k）
        if pfa_indices[shift] is None:
            attn = (q @ k.transpose(-2, -1))  # 标准注意力计算
            # 添加相对位置偏置
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
            attn = attn + relative_position_bias
            # 应用 SW-MSA 的窗口移位遮罩（仅在 shift=1 时生效）
            if shift:
                nw = mask.shape[0]
                attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
                attn = attn.view(-1, self.num_heads, n, n)
        else:
            # 稀疏注意力路径：仅计算被上一层保留下来的 top-k 位置，显著降低计算复杂度
            topk = pfa_indices[shift].shape[-1]
            qv = q.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            kv = k.contiguous().view(b_ * self.num_heads, n, c // self.num_heads).transpose(-2, -1)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            attn = SMM_QmK.apply(qv, kv, smm_index).view(b_, self.num_heads, n, topk)

            # 为稀疏注意力添加相对位置偏置（需要先扩展再 gather）
            relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
                self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0).expand(
                b_, self.num_heads, n, n
            )
            relative_position_bias = torch.gather(relative_position_bias, dim=-1, index=pfa_indices[shift])
            attn = attn + relative_position_bias

        attn = self.softmax(attn)  # Softmax 归一化

        # 渐进式聚焦：使用上一层保存的注意力值对当前注意力做 Hadamard 重加权并归一化
        if pfa_values[shift] is not None:
            attn = (attn * pfa_values[shift])  # Hadamard 乘积
            attn = (attn + self.eps) / (attn.sum(dim=-1, keepdim=True) + self.eps)  # 重新归一化

        # 本层的稀疏化：按行选取 top-k，并更新索引以供下一层复用
        if self.topk < self.window_size[0] * self.window_size[1]:
            topk_values, topk_indices = torch.topk(attn, self.topk, dim=-1, largest=True, sorted=False)
            attn = topk_values
            if pfa_indices[shift] is not None:
                # 如果已有索引，则从现有索引中进一步筛选
                pfa_indices[shift] = torch.gather(pfa_indices[shift], dim=-1, index=topk_indices)
            else:
                # 第一层，直接使用新的索引
                pfa_indices[shift] = topk_indices

        pfa_values[shift] = attn  # 保存当前层的注意力值供下一层使用

        # 根据是否稀疏，选择 A@V 的计算路径：密集矩阵乘法或 CUDA 稀疏乘法（带 CPU 回退）
        if pfa_indices[shift] is None:
            # 密集路径：标准矩阵乘法
            x = ((attn @ v) + v_lepe).transpose(1, 2).reshape(b_, n, c)
        else:
            # 稀疏路径：使用稀疏矩阵乘法
            topk = pfa_indices[shift].shape[-1]
            attn_ = attn.view(b_ * self.num_heads, n, topk)
            v_ = v.contiguous().view(b_ * self.num_heads, n, c // self.num_heads)
            smm_index = pfa_indices[shift].view(b_ * self.num_heads, n, topk).int()
            x = (SMM_AmV.apply(attn_, v_, smm_index).view(b_, self.num_heads, n, c // self.num_heads) + v_lepe).transpose(1, 2).reshape(b_, n, c)

        x = self.proj(x)  # 输出投影
        return x, pfa_values, pfa_indices


def _demo_main():
    """最小化自测试（使用随机张量）。

    运行方式:
        conda activate torchv5
        python pfa_module.py
    """
    torch.manual_seed(0)
    window_size = 8
    dim = 64
    num_heads = 8
    layer_id = 0
    n = window_size * window_size
    b_times_nw = 2  # 窗口数 * 批次大小

    # 每层的 top-k 调度（示例保持全量，便于快速验证数值正确性）
    num_topk = [n] * 24

    # 相对位置索引计算（与 PFT 的 calculate_rpi_sa 等价）
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += window_size - 1
    relative_coords[:, :, 1] += window_size - 1
    relative_coords[:, :, 0] *= 2 * window_size - 1
    rpi = relative_coords.sum(-1)

    # 生成假的 q,k,v 和局部位置编码 v (lepe)
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


