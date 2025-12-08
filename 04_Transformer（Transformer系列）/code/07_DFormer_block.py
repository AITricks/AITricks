"""
即插即用的几何先验 + 几何自注意力模块（从 DFormerv2 抽取）

主要包含三个部分：
- GeoPriorGen: 根据深度图和空间位置生成几何先验（几何 prior）
- Full_GSA / Decomposed_GSA: 几何自注意力（Geometry Self-Attention）
- RGBD_Block: 将几何先验和自注意力结合的基本 RGB-D Block，可直接插入到网络中

输入/输出约定（与原文一致）：
- 主干特征 x: 形状为 (B, H, W, C)
- 深度图 x_e: 形状为 (B, 1, H, W)
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class DWConv2d(nn.Module):
    """深度可分离卷积，用于局部位置编码（LePE）等。

    输入: (B, H, W, C)
    输出: (B, H, W, C)
    """

    def __init__(self, dim: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim)  # 深度可分离卷积：每个通道独立卷积

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b h w c) -> (b c h w)
        x = x.permute(0, 3, 1, 2)  # 调整维度以匹配 Conv2d 的 (B,C,H,W)
        x = self.dwconv(x)  # 逐通道卷积提取局部位置特征
        # (b c h w) -> (b h w c)
        x = x.permute(0, 2, 3, 1)  # 恢复到 (B,H,W,C)
        return x


def angle_transform(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """对 q/k 做旋转位置编码（RoPE 形式）。
    
    使用旋转位置编码（Rotary Position Embedding, RoPE）对查询和键向量进行位置编码。
    通过旋转操作将位置信息编码到特征向量中。
    
    Args:
        x: 输入张量，形状为 (b, n_head, H, W, d)，其中 d 是特征维度
        sin: 正弦位置编码，形状为 (H, W, d//2)
        cos: 余弦位置编码，形状为 (H, W, d//2)
    
    Returns:
        旋转编码后的张量，形状与输入相同
    """
    # x: (b, n_head, H, W, d)
    x1 = x[:, :, :, :, ::2]  # 提取偶数位置的特征维度
    x2 = x[:, :, :, :, 1::2]  # 提取奇数位置的特征维度
    # 应用旋转位置编码：x*cos + [-x2, x1]*sin
    return (x * cos) + (torch.stack([-x2, x1], dim=-1).flatten(-2) * sin)  # 旋转位置编码融合


class GeoPriorGen(nn.Module):
    """几何先验生成模块（从深度 + 空间位置得到几何 prior）。

    - 支持完整 2D 几何先验，或拆分为行/列的一维几何先验。
    """

    def __init__(self, embed_dim: int, num_heads: int, initial_value: float, heads_range: float):
        """初始化几何先验生成器。
        
        Args:
            embed_dim: 嵌入维度（通道数）
            num_heads: 注意力头数
            initial_value: 几何衰减的初始值
            heads_range: 不同注意力头之间的衰减范围
        """
        super().__init__()
        # 计算旋转位置编码（RoPE）的频率基，用于生成位置编码
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))  # RoPE 频率基
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()  # 扩展为奇偶交错频率，匹配特征维度
        # 可学习的权重，用于融合位置先验和深度先验：[位置权重, 深度权重]
        self.weight = nn.Parameter(torch.ones(2, 1, 1, 1), requires_grad=True)  # 可学习的先验融合权重 [位置, 深度]
        # 为不同注意力头计算递减的衰减率，距离越远的像素衰减越大
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads)
        )  # 为不同注意力头设置递减的衰减率
        self.register_buffer("angle", angle)  # 缓存 RoPE 频率，不参与梯度更新
        self.register_buffer("decay", decay)  # 缓存衰减率参数，不参与梯度更新

    def generate_depth_decay(self, H: int, W: int, depth_grid: torch.Tensor) -> torch.Tensor:
        """生成 2D 深度差衰减 mask，形状 (B, num_heads, HW, HW)。
        
        根据深度图计算像素之间的深度差异，生成衰减掩码。
        深度差异越大，衰减越大，使得深度相近的像素更容易相互关注。
        
        Args:
            H: 特征图高度
            W: 特征图宽度
            depth_grid: 深度图，形状为 (B, 1, H, W)
        
        Returns:
            深度衰减掩码，形状为 (B, num_heads, HW, HW)
        """
        B, _, H, W = depth_grid.shape
        grid_d = depth_grid.reshape(B, H * W, 1)  # 展平成向量以计算两两差值
        mask_d = grid_d[:, :, None, :] - grid_d[:, None, :, :]  # 计算所有像素对之间的深度差矩阵
        mask_d = (mask_d.abs()).sum(dim=-1)  # 使用 L1 距离度量深度差异
        mask_d = mask_d.unsqueeze(1) * self.decay[None, :, None, None]  # 乘以按头数设置的衰减系数
        return mask_d

    def generate_pos_decay(self, H: int, W: int) -> torch.Tensor:
        """生成 2D 位置距离衰减 mask，形状 (num_heads, HW, HW)。
        
        根据像素之间的空间距离生成衰减掩码。
        距离越远的像素衰减越大，使得空间上相近的像素更容易相互关注。
        
        Args:
            H: 特征图高度
            W: 特征图宽度
        
        Returns:
            位置衰减掩码，形状为 (num_heads, HW, HW)
        """
        index_h = torch.arange(H).to(self.decay)  # 行索引
        index_w = torch.arange(W).to(self.decay)  # 列索引
        grid = torch.meshgrid([index_h, index_w], indexing='ij')  # 构造网格坐标
        grid = torch.stack(grid, dim=-1).reshape(H * W, 2)  # 展平到 (HW,2)，每个元素是 (h, w) 坐标
        mask = grid[:, None, :] - grid[None, :, :]  # 计算所有像素对之间的坐标差
        mask = (mask.abs()).sum(dim=-1)  # 使用曼哈顿距离（L1 距离）度量空间距离
        mask = mask * self.decay[:, None, None]  # 乘以按头数设置的衰减系数
        return mask

    def generate_1d_depth_decay(self, H: int, W: int, depth_grid: torch.Tensor) -> torch.Tensor:
        """生成 1D 深度衰减 mask，用于分解式注意力。
        
        在分解式注意力中，分别沿行和列方向计算深度差异。
        这样可以降低计算复杂度，同时保持几何先验的有效性。
        
        Args:
            H: 特征图高度
            W: 特征图宽度
            depth_grid: 深度图，形状为 (B, 1, H, W) 或 (B, 1, W, H)
        
        Returns:
            1D 深度衰减掩码，形状为 (num_heads, B, W, H, H) 或类似
        """
        mask = depth_grid[:, :, :, :, None] - depth_grid[:, :, :, None, :]  # 计算 1D 方向上的深度差
        mask = mask.abs()  # 取绝对值作为距离度量
        mask = mask * self.decay[:, None, None, None]  # 应用按头数设置的衰减系数
        assert mask.shape[2:] == (W, H, H)  # 验证输出形状
        return mask

    def generate_1d_decay(self, l: int) -> torch.Tensor:
        """生成 1D 位置衰减 mask，用于分解式注意力。
        
        在分解式注意力中，分别沿行和列方向计算位置距离。
        
        Args:
            l: 一维序列长度（H 或 W）
        
        Returns:
            1D 位置衰减掩码，形状为 (num_heads, l, l)
        """
        index = torch.arange(l).to(self.decay)  # 位置索引
        mask = index[:, None] - index[None, :]  # 计算所有位置对之间的距离差
        mask = mask.abs()  # 取绝对值作为距离度量
        mask = mask * self.decay[:, None, None]  # 乘以按头数设置的衰减系数
        return mask

    def forward(self, HW_tuple: Tuple[int, int], depth_map: torch.Tensor, split_or_not: bool = False):
        """生成几何先验。

        Args:
            HW_tuple: (H, W) 补丁网格大小。
            depth_map: (B, 1, H_d, W_d) 深度图（会插值到 H, W）。
            split_or_not: True 生成行/列分解的一维 prior，False 生成完整 2D prior。
        """
        H, W = HW_tuple  # 特征网格尺寸
        depth_map = F.interpolate(depth_map, size=(H, W), mode="bilinear", align_corners=False)  # 深度图插值到匹配尺寸

        index = torch.arange(H * W).to(self.decay)  # 位置索引
        sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)  # 正弦位置编码
        cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)  # 余弦位置编码

        if split_or_not:
            # 生成行/列的一维 prior
            mask_d_h = self.generate_1d_depth_decay(H, W, depth_map.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(W, H, depth_map)

            mask_h = self.generate_1d_decay(H)
            mask_w = self.generate_1d_decay(W)

            mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_h  # 行方向融合位置/深度先验
            mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + self.weight[1] * mask_d_w  # 列方向融合位置/深度先验

            geo_prior = ((sin, cos), (mask_h, mask_w))  # 返回行/列一维先验
        else:
            # 生成完整 2D prior
            mask = self.generate_pos_decay(H, W)
            mask_d = self.generate_depth_decay(H, W, depth_map)
            mask = self.weight[0] * mask + self.weight[1] * mask_d

            geo_prior = ((sin, cos), mask)

        return geo_prior


class Decomposed_GSA(nn.Module):
    """分解式几何自注意力（沿 H/W 两个方向分两次注意力）。"""

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        """初始化分解式几何自注意力模块。
        
        Args:
            embed_dim: 嵌入维度（通道数）
            num_heads: 注意力头数
            value_factor: 值向量的扩展因子，默认为 1
        """
        super().__init__()  # 调用父类构造
        self.factor = value_factor  # 值向量扩展因子
        self.embed_dim = embed_dim  # 通道维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = self.embed_dim * self.factor // num_heads  # 每头 value 维度
        self.key_dim = self.embed_dim // num_heads  # 每头 key/query 维度
        self.scaling = self.key_dim**-0.5  # 缩放以稳定点积范围
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # q 投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # k 投影
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)  # v 投影（乘以 factor）
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)  # 局部位置编码（深度可分离卷积）
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)  # 输出线性层
        self.reset_parameters()  # 参数初始化

    def forward(self, x: torch.Tensor, rel_pos, split_or_not: bool = False) -> torch.Tensor:
        """分解式几何自注意力的前向传播。
        
        将完整的 2D 注意力分解为沿宽度（W）和高度（H）两个方向的一维注意力，
        从而降低计算复杂度从 O(H²W²) 到 O(H²W + HW²)。
        
        Args:
            x: 输入特征，形状为 (B, H, W, C)
            rel_pos: 相对位置编码，包含 (sin, cos) 和 (mask_h, mask_w)
            split_or_not: 是否使用分解式注意力（此参数在此类中始终为 True）
        
        Returns:
            输出特征，形状为 (B, H, W, C)
        """
        bsz, h, w, _ = x.size()  # 获取批次大小和空间尺寸
        (sin, cos), (mask_h, mask_w) = rel_pos  # 解包相对位置编码和掩码

        # 计算查询（query）、键（key）和值（value）
        q = self.q_proj(x)  # 查询投影
        k = self.k_proj(x)  # 键投影
        v = self.v_proj(x)  # 值投影
        lepe = self.lepe(v)  # 局部位置编码（LePE）

        k = k * self.scaling  # 缩放键向量以稳定注意力计算
        # 重塑并重排维度：(B, H, W, C) -> (B, num_heads, H, W, key_dim)
        q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
        # 应用旋转位置编码（RoPE）
        qr = angle_transform(q, sin, cos)  # 旋转编码查询
        kr = angle_transform(k, sin, cos)  # 旋转编码键

        # 第一步：沿宽度（W）方向进行一维注意力
        qr_w = qr.transpose(1, 2)  # 重排维度以便沿 W 方向计算
        kr_w = kr.transpose(1, 2)
        v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)  # 重塑值向量
        qk_mat_w = qr_w @ kr_w.transpose(-1, -2)  # 计算注意力分数矩阵
        qk_mat_w = qk_mat_w + mask_w.transpose(1, 2)  # 加入宽度方向的几何先验掩码
        qk_mat_w = torch.softmax(qk_mat_w, -1)  # Softmax 归一化
        v = torch.matmul(qk_mat_w, v)  # 聚合值向量

        # 第二步：沿高度（H）方向进行一维注意力
        qr_h = qr.permute(0, 3, 1, 2, 4)  # 重排维度以便沿 H 方向计算
        kr_h = kr.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 2, 1, 4)  # 调整值向量维度
        qk_mat_h = qr_h @ kr_h.transpose(-1, -2)  # 计算注意力分数矩阵
        qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)  # 加入高度方向的几何先验掩码
        qk_mat_h = torch.softmax(qk_mat_h, -1)  # Softmax 归一化
        output = torch.matmul(qk_mat_h, v)  # 聚合值向量

        # 重塑输出并加上局部位置编码
        output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)  # 恢复到 (B, H, W, num_heads*head_dim)
        output = output + lepe  # 残差连接局部位置编码
        output = self.out_proj(output)  # 输出投影层
        return output

    def reset_parameters(self):
        """参数初始化（Xavier 正态分布）。
        
        使用 Xavier 正态分布初始化线性层的权重，有助于训练稳定性。
        """
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)  # 查询投影层权重初始化
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)  # 键投影层权重初始化
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)  # 值投影层权重初始化
        nn.init.xavier_normal_(self.out_proj.weight)  # 输出投影层权重初始化
        nn.init.constant_(self.out_proj.bias, 0.0)  # 输出层偏置初始化为零


class Full_GSA(nn.Module):  # 完整二维几何自注意力模块
    """完整 2D 几何自注意力。"""

    def __init__(self, embed_dim: int, num_heads: int, value_factor: int = 1):
        """初始化完整 2D 几何自注意力模块。
        
        Args:
            embed_dim: 嵌入维度（通道数）
            num_heads: 注意力头数
            value_factor: 值向量的扩展因子，默认为 1
        """
        super().__init__()
        self.factor = value_factor  # 值向量扩展因子
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = self.embed_dim * self.factor // num_heads  # 每个注意力头的值向量维度
        self.key_dim = self.embed_dim // num_heads  # 每个注意力头的键/查询向量维度
        self.scaling = self.key_dim**-0.5  # 缩放因子，用于稳定注意力计算
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 查询投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)  # 键投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim * self.factor, bias=True)  # 值投影层（可扩展）
        self.lepe = DWConv2d(embed_dim, 5, 1, 2)  # 局部位置编码（LePE），使用深度可分离卷积
        self.out_proj = nn.Linear(embed_dim * self.factor, embed_dim, bias=True)  # 输出投影层
        self.reset_parameters()  # 初始化参数

    def forward(self, x: torch.Tensor, rel_pos, split_or_not: bool = False) -> torch.Tensor:
        """完整 2D 几何自注意力的前向传播。
        
        在完整的 2D 空间上进行自注意力计算，同时融入几何先验信息。
        计算复杂度为 O(H²W²)，适用于较小的特征图。
        
        Args:
            x: 输入特征，形状为 (B, H, W, C)
            rel_pos: 相对位置编码，包含 (sin, cos) 和 2D 几何先验掩码
            split_or_not: 是否使用分解式注意力（此参数在此类中始终为 False）
        
        Returns:
            输出特征，形状为 (B, H, W, C)
        """
        bsz, h, w, _ = x.size()  # 获取批次大小和空间尺寸
        (sin, cos), mask = rel_pos  # 解包相对位置编码和 2D 几何先验掩码
        assert h * w == mask.size(3)  # 验证掩码维度是否匹配

        # 计算查询（query）、键（key）和值（value）
        q = self.q_proj(x)  # 查询投影
        k = self.k_proj(x)  # 键投影
        v = self.v_proj(x)  # 值投影
        lepe = self.lepe(v)  # 局部位置编码（LePE）

        k = k * self.scaling  # 缩放键向量以稳定注意力计算
        # 重塑并重排维度：(B, H, W, C) -> (B, num_heads, H, W, key_dim)
        q = q.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # 重排到按头维度
        k = k.view(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4)  # 与 q 同形状
        # 应用旋转位置编码（RoPE）
        qr = angle_transform(q, sin, cos)  # 对查询应用旋转位置编码
        kr = angle_transform(k, sin, cos)  # 对键应用旋转位置编码

        # 展平空间维度，将 2D 特征图转换为序列
        qr = qr.flatten(2, 3)  # 展平到序列维 (HW)，形状变为 (B, num_heads, HW, key_dim)
        kr = kr.flatten(2, 3)  # 与 qr 对齐
        vr = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 3, 1, 2, 4).flatten(2, 3)  # 展平值向量

        # 计算完整的 2D 注意力
        qk_mat = qr @ kr.transpose(-1, -2)  # 点积注意力分数矩阵，形状为 (B, num_heads, HW, HW)
        qk_mat = qk_mat + mask  # 加入 2D 几何先验掩码（位置和深度先验）
        qk_mat = torch.softmax(qk_mat, -1)  # Softmax 归一化得到注意力权重
        output = torch.matmul(qk_mat, vr)  # 使用注意力权重聚合值向量

        # 恢复空间维度并加上局部位置编码
        output = output.transpose(1, 2).reshape(bsz, h, w, -1)  # 恢复到 (B, H, W, num_heads*head_dim)
        output = output + lepe  # 残差连接局部位置编码
        output = self.out_proj(output)  # 输出投影层，恢复到 embed_dim
        return output  # 返回注意力输出

    def reset_parameters(self):
        """参数初始化（Xavier 正态分布）。
        
        使用 Xavier 正态分布初始化线性层的权重，有助于训练稳定性。
        """
        nn.init.xavier_normal_(self.q_proj.weight, gain=2**-2.5)  # 查询投影层权重初始化
        nn.init.xavier_normal_(self.k_proj.weight, gain=2**-2.5)  # 键投影层权重初始化
        nn.init.xavier_normal_(self.v_proj.weight, gain=2**-2.5)  # 值投影层权重初始化
        nn.init.xavier_normal_(self.out_proj.weight)  # 输出投影层权重初始化
        nn.init.constant_(self.out_proj.bias, 0.0)  # 输出层偏置初始化为零


class FeedForwardNetwork(nn.Module):
    """简化版 FFN，与 DFormerv2 中对应部分一致。"""

    def __init__(
        self,
        embed_dim: int,  # 输入嵌入维度
        ffn_dim: int,  # 中间扩展维度
        activation_fn=F.gelu,  # 激活函数
        dropout: float = 0.0,  # 输出层 Dropout 比例
        activation_dropout: float = 0.0,  # 激活后 Dropout 比例
        layernorm_eps: float = 1e-6,  # LayerNorm eps
        subln: bool = False,  # 是否在中间使用 LayerNorm
        subconv: bool = True,  # 是否在中间使用 DWConv
    ):  # FFN 初始化
        super().__init__()  # 调用父类构造
        self.embed_dim = embed_dim  # 记录嵌入维度
        self.activation_fn = activation_fn  # 激活函数句柄
        self.activation_dropout_module = nn.Dropout(activation_dropout)  # 激活后 Dropout
        self.dropout_module = nn.Dropout(dropout)  # 输出 Dropout
        self.fc1 = nn.Linear(self.embed_dim, ffn_dim)  # 第一层线性扩展
        self.fc2 = nn.Linear(ffn_dim, self.embed_dim)  # 第二层线性回收
        self.ffn_layernorm = nn.LayerNorm(ffn_dim, eps=layernorm_eps) if subln else None  # 可选中间 LN
        self.dwconv = DWConv2d(ffn_dim, 3, 1, 1) if subconv else None  # 可选 DWConv 增强局部性

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # FFN 前向
        x = self.fc1(x)  # 线性扩展
        x = self.activation_fn(x)  # 激活
        x = self.activation_dropout_module(x)  # 激活后 Dropout
        residual = x  # 残差支路
        if self.dwconv is not None:
            x = self.dwconv(x)  # 局部卷积增强
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)  # 中间 LayerNorm
        x = x + residual  # 残差连接
        x = self.fc2(x)  # 线性回收到嵌入维度
        x = self.dropout_module(x)  # 输出 Dropout
        return x  # 返回输出


class RGBD_Block(nn.Module):  # 集成几何先验与自注意力的 RGB-D 模块
    """即插即用的 RGB-D 几何自注意力 Block。

    - 输入: x (B, H, W, C)、x_e (B, 1, H, W)
    - 输出: x (B, H, W, C)
    """

    def __init__(
        self,
        split_or_not: bool,  # 是否使用分解式注意力
        embed_dim: int,  # 嵌入通道维度
        num_heads: int,  # 注意力头数
        ffn_dim: int,  # FFN 扩展维度
        drop_path: float = 0.0,  # 随机深度比例
        layerscale: bool = False,  # 是否启用 LayerScale
        layer_init_values: float = 1e-5,  # LayerScale 初始化因子
        init_value: float = 2.0,  # 几何衰减初值
        heads_range: float = 4.0,  # 头间衰减范围
    ):  # 模块初始化
        super().__init__()  # 调用父类构造
        self.layerscale = layerscale  # 记录是否启用 LayerScale
        self.embed_dim = embed_dim  # 嵌入维度
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=1e-6)  # 注意力前 LN
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=1e-6)  # FFN 前 LN
        if split_or_not:
            self.Attention = Decomposed_GSA(embed_dim, num_heads)  # 分解式几何自注意力
        else:
            self.Attention = Full_GSA(embed_dim, num_heads)  # 完整几何自注意力
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()  # 随机深度
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)  # 前馈网络
        self.cnn_pos_encode = DWConv2d(embed_dim, 3, 1, 1)  # 卷积位置编码
        self.Geo = GeoPriorGen(embed_dim, num_heads, init_value, heads_range)  # 几何先验生成器

        if layerscale:  # 启用 LayerScale 时，添加可学习缩放因子
            self.gamma_1 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)  # 注意力分支缩放
            self.gamma_2 = nn.Parameter(layer_init_values * torch.ones(1, 1, 1, embed_dim), requires_grad=True)  # FFN 分支缩放

    def forward(self, x: torch.Tensor, x_e: torch.Tensor, split_or_not: bool = False) -> torch.Tensor:  # 模块前向
        # x: (B, H, W, C), x_e: (B, 1, H, W)
        x = x + self.cnn_pos_encode(x)  # 加入卷积位置编码
        b, h, w, d = x.size()  # 提取尺寸信息

        geo_prior = self.Geo((h, w), x_e, split_or_not=split_or_not)  # 生成几何先验（行/列或 2D）
        if self.layerscale:  # LayerScale 路径
            x = x + self.drop_path(self.gamma_1 * self.Attention(self.layer_norm1(x), geo_prior, split_or_not))  # 注意力分支
            x = x + self.drop_path(self.gamma_2 * self.ffn(self.layer_norm2(x)))  # FFN 分支
        else:  # 标准路径
            x = x + self.drop_path(self.Attention(self.layer_norm1(x), geo_prior, split_or_not))  # 注意力分支
            x = x + self.drop_path(self.ffn(self.layer_norm2(x)))  # FFN 分支
        return x  # 返回输出


def _demo_main():
    """简单用例演示：使用随机输入测试 RGBD_Block 的前向传播。
    
    创建一个 RGBD_Block 模块，使用随机生成的 RGB 特征和深度图进行前向推理，
    验证模块的基本功能是否正常。
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动选择设备（GPU 优先）
    B, C, H, W = 1, 3, 64, 64  # 批次大小和图像尺寸
    embed_dim = 64  # 嵌入维度（通道数）
    num_heads = 4  # 注意力头数
    ffn_dim = 4 * embed_dim  # FFN 扩展维度（通常为嵌入维度的 4 倍）
    # 假设 RGB 特征已经通过主干网络升维到 embed_dim
    rgb = torch.randn(B, embed_dim, H, W, device=device)  # 随机生成 RGB 特征，形状为 (B, C, H, W)
    depth = torch.randn(B, 1, H, W, device=device)  # 随机生成深度图，形状为 (B, 1, H, W)

    # 将特征转换为模块所需的格式 (B, H, W, C)
    x = rgb.permute(0, 2, 3, 1).contiguous()  # 调整为模块输入布局
    # 创建 RGBD_Block 模块
    block = RGBD_Block(
        split_or_not=False,  # 使用完整几何自注意力（而非分解式）
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        drop_path=0.0,  # 不使用随机深度
        layerscale=False,  # 不使用 LayerScale
    ).to(device)  # 将模块移动到指定设备

    with torch.no_grad():  # 评估模式，不计算梯度以节省内存
        y = block(x, depth, split_or_not=False)  # 前向推理
    print(f"输入形状: x={x.shape}, depth={depth.shape}")  # 打印输入形状
    print(f"输出形状: y={y.shape}")  # 打印输出形状

if __name__ == "__main__":
    _demo_main()


