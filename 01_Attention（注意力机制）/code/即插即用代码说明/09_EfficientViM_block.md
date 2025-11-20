# EfficientViM 即插即用模块说明文档

## 📋 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [模块说明](#模块说明)
- [安装与依赖](#安装与依赖)
- [使用方法](#使用方法)
- [性能特点](#性能特点)
- [示例代码](#示例代码)
- [API 参考](#api-参考)

---

## 概述

`EfficientViM_PlugAndPlay.py` 提供了 EfficientViM 模型中的核心模块作为即插即用组件，可以直接替换到其他 Vision Transformer 模型中。主要包含：

- **HSM-SSD 层**：Hidden State Memory - State Space Discretization 层，优化了计算复杂度的注意力机制
- **EfficientViMBlock**：完整的即插即用块，包含 HSM-SSD 和必要的辅助组件

### 核心优势

✅ **即插即用**：可以直接替换 ViT、Swin 等模型中的注意力层  
✅ **计算高效**：将 O(LD²) 操作优化为 O(ND²)，显著降低计算复杂度  
✅ **内存友好**：通过隐藏状态记忆机制减少内存占用  
✅ **易于集成**：提供完整的前向传播接口，支持多种输入格式

---

## 架构设计

### EfficientViM 整体架构

```
Input (H × W × 3)
    ↓
Stem Layer
    ↓
Stage 1: Efficient ViM Block × l1  →  (H/16) × (W/16) × D1  ──┐
    ↓                                                                  │
Patch Merging                                                            │
    ↓                                                                  │
Stage 2: Efficient ViM Block × l2  →  (H/32) × (W/32) × D2  ──┤ MSF
    ↓                                                                  │ (Multi-Stage
Patch Merging                                                         │  Fusion)
    ↓                                                                  │
Stage 3: Efficient ViM Block × l3  →  (H/64) × (W/64) × D3  ──┘
    ↓
Output
```

### EfficientViM Block 内部结构

```
Input (B, C, H, W)
    ↓
┌─────────────────────────────────────┐
│  3×3 DWConv + Residual              │
├─────────────────────────────────────┤
│  LayerNorm                           │
│    ↓                                 │
│  HSM-SSD (核心注意力模块)            │
│    ↓                                 │
│  Residual                            │
├─────────────────────────────────────┤
│  3×3 DWConv + Residual              │
├─────────────────────────────────────┤
│  FFN (Feed-Forward Network)         │
│    ↓                                 │
│  Residual                            │
└─────────────────────────────────────┘
Output (B, C, H, W), Hidden State (B, C, N)
```

### HSM-SSD 层详细设计

HSM-SSD 层是核心创新，优化了传统的 NC-SSD 层：

#### NC-SSD (原始版本)
- **复杂度**: O(LD²) - 其中 L 是序列长度，D 是特征维度
- **问题**: 当 L 很大时（高分辨率图像），计算开销巨大

#### HSM-SSD (优化版本)
- **复杂度**: O(ND²) - 其中 N 是隐藏状态维度（通常 N << L）
- **优化**: 将计算从序列空间转移到隐藏状态空间
- **优势**: 计算复杂度与序列长度 L 无关，仅与隐藏状态维度 N 相关

**HSM-SSD 数据流**：

```
Input (B, C, L)
    ↓
BCdt_proj: (B, C, L) → (B, 3N, L)
    ↓
DWConv (空间卷积)
    ↓
Split → B, C, dt (每个 N 维)
    ↓
Discretization: A = softmax(dt + A_init)
    ↓
State Space: h = x @ (A * B)^T  (B, C, N)
    ↓
HSM Component (O(ND²) 计算) ──┐
    ↓                          │ Orange Box
h = FFN(h, z)                  │
    ↓                          │
Output: y = h @ C              │
    ↓                          │
(B, C, H, W), (B, C, N)        │
```

---

## 模块说明

### 1. HSMSSD 类

核心的 Hidden State Memory - State Space Discretization 层。

#### 初始化参数

```python
HSMSSD(
    d_model,           # 模型维度 (int)
    ssd_expand=1,      # SSD 扩展因子 (float, 默认 1)
    A_init_range=(1, 16),  # A 参数初始化范围 (tuple, 默认 (1, 16))
    state_dim=64       # 隐藏状态维度 (int, 默认 64)
)
```

#### 输入输出

- **输入**: `(B, C, L)` - 其中 L = H × W
- **输出**: 
  - `y`: `(B, C, H, W)` - 输出特征图
  - `h`: `(B, C, N)` - 隐藏状态（可用于多阶段融合）

#### 关键特性

- ✅ 计算复杂度 O(ND²)，与序列长度 L 无关
- ✅ 支持空间感知的深度卷积
- ✅ 可学习的残差连接权重

### 2. EfficientViMBlock 类

完整的即插即用块，包含所有必要组件。

#### 初始化参数

```python
EfficientViMBlock(
    dim,               # 特征维度 (int)
    mlp_ratio=4.,      # MLP 扩展比例 (float, 默认 4.0)
    ssd_expand=1,      # SSD 扩展因子 (float, 默认 1)
    state_dim=64       # 隐藏状态维度 (int, 默认 64)
)
```

#### 输入输出

- **输入**: `(B, C, H, W)` - 2D 特征图
- **输出**: 
  - `x`: `(B, C, H, W)` - 输出特征图
  - `h`: `(B, C, N)` - 隐藏状态

#### 组件说明

1. **DWConv1**: 3×3 深度卷积，提取局部特征
2. **HSM-SSD**: 核心注意力模块，捕获长距离依赖
3. **DWConv2**: 另一个 3×3 深度卷积，进一步处理特征
4. **FFN**: 前馈网络，扩展特征维度
5. **LayerScale**: 每个组件都有可学习的残差权重，确保训练稳定性

---

## 安装与依赖

### 依赖要求

```python
torch >= 1.8.0
torch.nn
```

### 导入方式

```python
from EfficientViM_PlugAndPlay import HSMSSD, EfficientViMBlock
# 或者
from classification.models.EfficientViM_PlugAndPlay import HSMSSD, EfficientViMBlock
```

---

## 使用方法

### 基础使用

#### 1. 使用 HSMSSD 替换注意力层

```python
import torch
import torch.nn as nn
from EfficientViM_PlugAndPlay import HSMSSD

# 创建 HSM-SSD 模块
d_model = 256
state_dim = 64
hsm_ssd = HSMSSD(d_model=d_model, state_dim=state_dim)

# 输入: (B, C, L) 其中 L = H * W
batch_size = 4
H, W = 14, 14
x = torch.randn(batch_size, d_model, H * W)

# 前向传播
y, h = hsm_ssd(x)  # y: (B, C, H, W), h: (B, C, N)

print(f"输出特征形状: {y.shape}")
print(f"隐藏状态形状: {h.shape}")
```

#### 2. 使用 EfficientViMBlock 替换 Transformer Block

```python
import torch
from EfficientViM_PlugAndPlay import EfficientViMBlock

# 创建 Block
dim = 256
block = EfficientViMBlock(dim=dim, mlp_ratio=4.0, state_dim=64)

# 输入: (B, C, H, W)
batch_size = 4
H, W = 14, 14
x = torch.randn(batch_size, dim, H, W)

# 前向传播
x_out, h = block(x)

print(f"输出形状: {x_out.shape}")
print(f"隐藏状态形状: {h.shape}")
```

### 集成到现有模型

#### 示例 1: 替换 ViT 中的注意力层

```python
import torch
import torch.nn as nn
from EfficientViM_PlugAndPlay import HSMSSD, LayerNorm1D

class ViTBlockWithHSMSSD(nn.Module):
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.norm1 = LayerNorm1D(dim)
        self.attention = HSMSSD(d_model=dim, state_dim=state_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
    def forward(self, x):
        # x: (B, N, C) -> (B, C, N)
        B, N, C = x.shape
        x_reshaped = x.permute(0, 2, 1)  # (B, C, N)
        
        # HSM-SSD attention
        x_attn, h = self.attention(self.norm1(x_reshaped))
        x_attn = x_attn.flatten(2).permute(0, 2, 1)  # (B, N, C)
        
        x = x + x_attn
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x
```

#### 示例 2: 替换 Swin Transformer Block

```python
import torch
import torch.nn as nn
from EfficientViM_PlugAndPlay import EfficientViMBlock

class SwinBlockWithEfficientViM(nn.Module):
    def __init__(self, dim, state_dim=64):
        super().__init__()
        # 直接使用 EfficientViMBlock
        self.block = EfficientViMBlock(dim=dim, state_dim=state_dim)
        
    def forward(self, x):
        # x: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x, h = self.block(x)
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        return x
```

#### 示例 3: 构建完整的多阶段模型

```python
import torch
import torch.nn as nn
from EfficientViM_PlugAndPlay import EfficientViMBlock

class MultiStageEfficientViM(nn.Module):
    def __init__(self, embed_dims=[128, 256, 512], depths=[2, 2, 2], state_dims=[49, 25, 9]):
        super().__init__()
        self.stages = nn.ModuleList()
        
        for i, (dim, depth, state_dim) in enumerate(zip(embed_dims, depths, state_dims)):
            stage = nn.Sequential(*[
                EfficientViMBlock(dim=dim, state_dim=state_dim)
                for _ in range(depth)
            ])
            self.stages.append(stage)
            
        # 多阶段隐藏状态融合
        self.fusion_weights = nn.Parameter(torch.ones(len(embed_dims)))
        
    def forward(self, x):
        hidden_states = []
        
        for stage in self.stages:
            x, h = stage[0](x)  # 简化示例，实际需要处理多个 blocks
            hidden_states.append(h)
            
        # 加权融合隐藏状态
        weights = self.fusion_weights.softmax(-1)
        fused_hidden = sum(w * h for w, h in zip(weights, hidden_states))
        
        return x, fused_hidden
```

---

## 性能特点

### 计算复杂度对比

| 操作 | NC-SSD (原始) | HSM-SSD (优化) | 改进 |
|------|--------------|----------------|------|
| 投影层 | O(LD²) | O(LND) | 降低 |
| HSM 组件 | - | O(ND²) | 新增（高效） |
| 矩阵乘法 | O(LND) | O(LND) | 相同 |
| **总体** | **O(LD²)** | **O(ND²)** | **显著降低** |

### 优势说明

1. **计算效率**: 当 L (序列长度) >> N (隐藏状态维度) 时，复杂度从 O(LD²) 降低到 O(ND²)
   - 例如: L=196 (14×14), D=256, N=64
   - 原始: O(196 × 256²) ≈ 12.8M
   - 优化: O(64 × 256²) ≈ 4.2M
   - **提升约 3 倍**

2. **内存效率**: 隐藏状态维度 N 远小于序列长度 L，减少内存占用

3. **空间感知**: 通过深度卷积保持空间结构信息

4. **训练稳定**: LayerScale 机制确保稳定的梯度流

---

## 示例代码

### 完整测试示例

运行测试代码：

```bash
python classification/models/EfficientViM_PlugAndPlay.py
```

测试包括：
- ✅ HSMSSD 模块测试
- ✅ EfficientViMBlock 模块测试  
- ✅ 即插即用使用示例

### 自定义测试

```python
import torch
from EfficientViM_PlugAndPlay import EfficientViMBlock

# 创建模型
model = EfficientViMBlock(dim=256, state_dim=64)

# 创建输入
x = torch.randn(2, 256, 14, 14)

# 前向传播
with torch.no_grad():
    x_out, h = model(x)
    
print(f"输入: {x.shape}")
print(f"输出: {x_out.shape}")
print(f"隐藏状态: {h.shape}")

# 计算参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
```

---

## API 参考

### HSMSSD

```python
class HSMSSD(nn.Module):
    """
    Hidden State Memory - State Space Discretization 层
    
    Args:
        d_model (int): 模型维度
        ssd_expand (float): SSD 扩展因子，默认 1
        A_init_range (tuple): A 参数初始化范围，默认 (1, 16)
        state_dim (int): 隐藏状态维度，默认 64
    
    Input:
        x (torch.Tensor): 形状为 (B, C, L) 的输入，其中 L = H × W
    
    Returns:
        y (torch.Tensor): 形状为 (B, C, H, W) 的输出特征
        h (torch.Tensor): 形状为 (B, C, N) 的隐藏状态
    """
```

### EfficientViMBlock

```python
class EfficientViMBlock(nn.Module):
    """
    EfficientViM Block - 完整的即插即用块
    
    Args:
        dim (int): 特征维度
        mlp_ratio (float): MLP 扩展比例，默认 4.0
        ssd_expand (float): SSD 扩展因子，默认 1
        state_dim (int): 隐藏状态维度，默认 64
    
    Input:
        x (torch.Tensor): 形状为 (B, C, H, W) 的输入特征图
    
    Returns:
        x (torch.Tensor): 形状为 (B, C, H, W) 的输出特征图
        h (torch.Tensor): 形状为 (B, C, N) 的隐藏状态
    """
```

### 工具类

- `LayerNorm1D`: 1D 张量的 LayerNorm（通道维度）
- `LayerNorm2D`: 2D 张量的 LayerNorm（通道维度）
- `ConvLayer1D`: 1D 卷积层（带归一化和激活）
- `ConvLayer2D`: 2D 卷积层（带归一化和激活）
- `FFN`: 前馈网络

---

## 注意事项

1. **输入格式**: 
   - `HSMSSD` 需要输入 `(B, C, L)` 格式，需要先将 2D 特征 flatten
   - `EfficientViMBlock` 直接接受 `(B, C, H, W)` 格式

2. **隐藏状态维度**: 
   - `state_dim` 通常设置为小于序列长度 L 的值
   - 推荐值: N = sqrt(L) 或更小（如 L=196 时，N=49 或 64）

3. **内存优化**: 
   - 如果遇到内存问题，可以减小 `ssd_expand` 或 `mlp_ratio`
   - 调整 `state_dim` 也可以平衡性能和内存

4. **训练建议**: 
   - 初始学习率可以设置为原始模型的 0.5-1.0 倍
   - LayerScale 的初始权重很小（1e-4），确保训练稳定

---

## 参考文献

- EfficientViM: Efficient Vision Mamba with Cross-Space State Discretization
- 原始实现: [EfficientViM GitHub](https://github.com/hkchengrex/EfficientViM)

---

## 许可证

请参考原始项目的 LICENSE 文件。

---

## 更新日志

### v1.0.0 (2024)
- ✅ 初始版本
- ✅ 实现 HSMSSD 和 EfficientViMBlock
- ✅ 添加完整的测试代码
- ✅ 提供详细的使用文档

---

**如有问题或建议，欢迎提出 Issue 或 Pull Request！**

