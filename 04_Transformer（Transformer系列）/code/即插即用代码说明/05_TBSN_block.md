# TBSN 即插即用模块说明文档

## 📋 目录

- [项目简介](#项目简介)
- [模块列表](#模块列表)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [模块详细说明](#模块详细说明)
- [使用示例](#使用示例)
- [测试](#测试)
- [架构说明](#架构说明)

---

## 项目简介

本项目从 TBSN (Transformer-Based Blind-Spot Network) 中提取了核心的即插即用模块，这些模块可以独立使用或集成到其他深度学习网络中。TBSN 是一个用于自监督图像去噪的 Transformer 架构，通过扩张卷积和掩码注意力机制实现了盲点网络的功能。

### 主要特性

- ✅ **即插即用**：所有模块都可以独立使用
- ✅ **完整测试**：所有模块都经过完整测试
- ✅ **详细文档**：每个模块都有清晰的说明
- ✅ **易于集成**：可以轻松集成到现有网络中

---

## 模块列表

### 核心注意力模块

1. **DilatedMDTA** - 扩张多头通道自注意力（对应结构图中的 Dilated G-CSA）
2. **DilatedOCA** - 扩张重叠交叉注意力（对应结构图中的 Dilated M-WSA）
3. **FeedForward** - 扩张前馈网络（对应结构图中的 Dilated FFN）
4. **TransformerBlock** - 完整的 Transformer 块（DTAB）

### 辅助模块

5. **LayerNorm** - 层归一化（支持有偏置/无偏置）
6. **CentralMaskedConv2d** - 中心掩码卷积（盲点网络核心）
7. **OverlapPatchEmbed** - 重叠 Patch 嵌入
8. **PatchUnshuffle** - Patch 下采样操作
9. **PatchShuffle** - Patch 上采样操作

### 位置编码模块

10. **RelPosEmb** - 相对位置编码
11. **FixedPosEmb** - 固定位置编码（用于掩码注意力）

---

## 安装要求

### Python 环境

- Python >= 3.8
- PyTorch >= 2.0.0
- einops

### 安装依赖

```bash
# 使用 conda 安装
conda create -n tbsn python=3.8
conda activate tbsn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install einops opencv-python

# 或使用 pip 安装
pip install torch torchvision einops
```

---

## 快速开始

### 基本使用

```python
import torch
from plug_module import (
    DilatedMDTA, DilatedOCA, FeedForward, TransformerBlock,
    OverlapPatchEmbed, CentralMaskedConv2d, LayerNorm
)

# 创建输入
x = torch.randn(2, 48, 64, 64)  # (batch, channels, height, width)

# 使用通道注意力模块
channel_attn = DilatedMDTA(dim=48, num_heads=2)
out = channel_attn(x)  # 输出形状: (2, 48, 64, 64)

# 使用空间注意力模块
spatial_attn = DilatedOCA(
    dim=48, 
    window_size=8, 
    overlap_ratio=0.5, 
    num_heads=2, 
    dim_head=16
)
out = spatial_attn(x)  # 输出形状: (2, 48, 64, 64)

# 使用完整的 Transformer 块
transformer = TransformerBlock(
    dim=48,
    window_size=8,
    overlap_ratio=0.5,
    num_channel_heads=2,
    num_spatial_heads=2,
    spatial_dim_head=16,
    ffn_expansion_factor=1,
    bias=False,
    LayerNorm_type='BiasFree'
)
out = transformer(x)  # 输出形状: (2, 48, 64, 64)
```

---

## 模块详细说明

### 1. DilatedMDTA (Dilated Multi-Head Channel-wise Self-Attention)

**对应结构图**: Dilated G-CSA (Grouped Channel-wise Self-Attention)

**功能**: 使用扩张卷积的通道注意力机制，实现通道间的自注意力计算。

**参数**:
- `dim` (int): 输入特征维度
- `num_heads` (int): 注意力头数
- `bias` (bool): 是否使用偏置，默认 False

**示例**:
```python
from plug_module import DilatedMDTA

# 创建模块
mdta = DilatedMDTA(dim=48, num_heads=2)

# 前向传播
x = torch.randn(2, 48, 64, 64)
out = mdta(x)  # 输出: (2, 48, 64, 64)
```

---

### 2. DilatedOCA (Dilated Overlapped Cross Attention)

**对应结构图**: Dilated M-WSA (Masked Window-based Self-Attention)

**功能**: 使用扩张卷积的窗口注意力机制，带有掩码，实现空间自注意力。

**参数**:
- `dim` (int): 输入特征维度
- `window_size` (int): 窗口大小（输入尺寸必须能被窗口大小整除）
- `overlap_ratio` (float): 重叠比例
- `num_heads` (int): 注意力头数
- `dim_head` (int): 每个头的维度
- `bias` (bool): 是否使用偏置，默认 False

**示例**:
```python
from plug_module import DilatedOCA

# 创建模块（输入尺寸必须是 window_size 的倍数）
oca = DilatedOCA(
    dim=48, 
    window_size=8, 
    overlap_ratio=0.5, 
    num_heads=2, 
    dim_head=16
)

# 前向传播（64x64 可以被 8 整除）
x = torch.randn(2, 48, 64, 64)
out = oca(x)  # 输出: (2, 48, 64, 64)
```

---

### 3. FeedForward (Dilated Feed-Forward Network)

**对应结构图**: Dilated FFN

**功能**: 使用扩张卷积的前馈网络，包含两个扩张卷积层和 GELU 激活。

**参数**:
- `dim` (int): 输入特征维度
- `ffn_expansion_factor` (float): 扩展因子（隐藏层维度 = dim * ffn_expansion_factor）
- `bias` (bool): 是否使用偏置，默认 False

**示例**:
```python
from plug_module import FeedForward

# 创建模块
ffn = FeedForward(dim=48, ffn_expansion_factor=1)

# 前向传播
x = torch.randn(2, 48, 64, 64)
out = ffn(x)  # 输出: (2, 48, 64, 64)
```

---

### 4. TransformerBlock (Dilated Transformer Attention Block)

**对应结构图**: DTAB (核心模块)

**功能**: 完整的 Transformer 块，组合了通道注意力、空间注意力和前馈网络。

**结构**:
```
输入 -> LayerNorm -> Channel Attention -> Residual
     -> LayerNorm -> Channel FFN -> Residual
     -> LayerNorm -> Spatial Attention -> Residual
     -> LayerNorm -> Spatial FFN -> Residual -> 输出
```

**参数**:
- `dim` (int): 输入特征维度
- `window_size` (int): 窗口大小
- `overlap_ratio` (float): 重叠比例
- `num_channel_heads` (int): 通道注意力头数
- `num_spatial_heads` (int): 空间注意力头数
- `spatial_dim_head` (int): 空间注意力每个头的维度
- `ffn_expansion_factor` (float): FFN 扩展因子
- `bias` (bool): 是否使用偏置
- `LayerNorm_type` (str): LayerNorm 类型，'BiasFree' 或 'WithBias'

**示例**:
```python
from plug_module import TransformerBlock

# 创建完整的 Transformer 块
transformer = TransformerBlock(
    dim=48,
    window_size=8,
    overlap_ratio=0.5,
    num_channel_heads=2,
    num_spatial_heads=2,
    spatial_dim_head=16,
    ffn_expansion_factor=1,
    bias=False,
    LayerNorm_type='BiasFree'
)

# 前向传播
x = torch.randn(2, 48, 64, 64)
out = transformer(x)  # 输出: (2, 48, 64, 64)
```

---

### 5. CentralMaskedConv2d

**功能**: 中心掩码卷积，用于盲点网络。中心像素的权重被置零，确保输出不依赖于中心输入。

**参数**: 与 `nn.Conv2d` 相同

**示例**:
```python
from plug_module import CentralMaskedConv2d

# 创建中心掩码卷积
conv = CentralMaskedConv2d(3, 48, kernel_size=3, padding=1)

# 前向传播
x = torch.randn(2, 3, 64, 64)
out = conv(x)  # 输出: (2, 48, 64, 64)
```

---

### 6. OverlapPatchEmbed

**功能**: 重叠 Patch 嵌入模块，使用中心掩码卷积将输入图像转换为特征图。

**参数**:
- `in_c` (int): 输入通道数，默认 3
- `embed_dim` (int): 嵌入维度，默认 48
- `bias` (bool): 是否使用偏置，默认 False

**示例**:
```python
from plug_module import OverlapPatchEmbed

# 创建 Patch 嵌入
patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=48)

# 前向传播
x = torch.randn(2, 3, 64, 64)
out = patch_embed(x)  # 输出: (2, 48, 64, 64)
```

---

### 7. PatchUnshuffle / PatchShuffle

**功能**: Patch 下采样和上采样操作，用于多尺度特征提取。

**参数**:
- `p` (int): 第一个下采样因子，默认 2
- `s` (int): 第二个下采样因子，默认 2

**示例**:
```python
from plug_module import PatchUnshuffle, PatchShuffle

# Patch 下采样
unshuffle = PatchUnshuffle(p=2, s=2)
x = torch.randn(2, 48, 64, 64)
down = unshuffle(x)  # 输出: (2, 192, 32, 32)

# Patch 上采样
shuffle = PatchShuffle(p=2, s=2)
up = shuffle(down)  # 输出: (2, 48, 64, 64)
```

---

### 8. LayerNorm

**功能**: 层归一化，支持有偏置和无偏置两种模式。

**参数**:
- `dim` (int): 特征维度
- `LayerNorm_type` (str): 'BiasFree' 或 'WithBias'，默认 'BiasFree'

**示例**:
```python
from plug_module import LayerNorm

# 创建无偏置 LayerNorm
norm = LayerNorm(dim=48, LayerNorm_type='BiasFree')

# 前向传播
x = torch.randn(2, 48, 64, 64)
out = norm(x)  # 输出: (2, 48, 64, 64)
```

---

## 使用示例

### 示例 1: 构建简单的去噪网络

```python
import torch
import torch.nn as nn
from plug_module import (
    OverlapPatchEmbed, TransformerBlock, 
    CentralMaskedConv2d
)

class SimpleDenoiser(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=48):
        super().__init__()
        # Patch 嵌入
        self.embed = OverlapPatchEmbed(in_c=in_ch, embed_dim=dim)
        
        # Transformer 块
        self.transformer = TransformerBlock(
            dim=dim,
            window_size=8,
            overlap_ratio=0.5,
            num_channel_heads=2,
            num_spatial_heads=2,
            spatial_dim_head=16,
            ffn_expansion_factor=1,
            bias=False,
            LayerNorm_type='BiasFree'
        )
        
        # 输出层
        self.output = nn.Conv2d(dim, out_ch, kernel_size=1)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        x = self.output(x)
        return x

# 使用
model = SimpleDenoiser()
x = torch.randn(2, 3, 64, 64)
out = model(x)  # 输出: (2, 3, 64, 64)
```

### 示例 2: 集成到现有网络

```python
import torch
import torch.nn as nn
from plug_module import DilatedMDTA, FeedForward, LayerNorm

class CustomBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm(dim, 'BiasFree')
        self.attn = DilatedMDTA(dim=dim, num_heads=2)
        self.ffn = FeedForward(dim=dim, ffn_expansion_factor=1)
    
    def forward(self, x):
        x = x + self.attn(self.norm(x))
        x = x + self.ffn(x)
        return x

# 集成到现有网络
class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, 3, padding=1)
        self.custom_block = CustomBlock(48)
        self.conv2 = nn.Conv2d(48, 3, 3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.custom_block(x)
        x = self.conv2(x)
        return x
```

---

## 测试

运行测试脚本验证所有模块：

```bash
# 激活 conda 环境
conda activate torchv5

# 运行测试
python plug_module.py
```

测试包括：
- ✅ LayerNorm
- ✅ CentralMaskedConv2d
- ✅ PatchUnshuffle / PatchShuffle
- ✅ OverlapPatchEmbed
- ✅ DilatedMDTA (Dilated G-CSA)
- ✅ DilatedOCA (Dilated M-WSA)
- ✅ FeedForward (Dilated FFN)
- ✅ TransformerBlock (DTAB)
- ✅ 位置编码模块
- ✅ 完整前向传播流程

---

## 架构说明

### TBSN 网络结构

TBSN 网络的核心是 **DTAB (Dilated Transformer Attention Block)**，它包含：

1. **Dilated G-CSA**: 分组通道自注意力，使用扩张卷积
2. **Dilated FFN**: 扩张前馈网络
3. **Dilated M-WSA**: 掩码窗口自注意力，使用扩张卷积和掩码

### 关键特性

- **盲点机制**: 通过 `CentralMaskedConv2d` 实现，中心像素不参与计算
- **扩张卷积**: 扩大感受野，同时保持计算效率
- **窗口注意力**: 使用固定窗口大小，降低计算复杂度
- **掩码机制**: 在窗口注意力中应用掩码，模拟扩张卷积的感受野

### 结构图对应关系

- **DilatedMDTA** ↔ 结构图中的 **Dilated G-CSA**
- **DilatedOCA** ↔ 结构图中的 **Dilated M-WSA**
- **FeedForward** ↔ 结构图中的 **Dilated FFN**
- **TransformerBlock** ↔ 结构图中的 **DTAB**

---

## 注意事项

1. **输入尺寸要求**: 
   - `DilatedOCA` 和 `TransformerBlock` 的输入高度和宽度必须能被 `window_size` 整除
   - 例如：如果 `window_size=8`，输入尺寸应为 64x64, 128x128 等

2. **设备兼容性**: 
   - 所有模块都支持 CPU 和 GPU
   - 使用 `.to(device)` 将模块移动到指定设备

3. **内存使用**: 
   - 窗口注意力模块的内存使用与窗口大小和输入尺寸相关
   - 对于大图像，建议使用较小的 `window_size`

4. **训练建议**: 
   - 建议使用 `LayerNorm_type='BiasFree'` 以获得更好的训练稳定性
   - 可以使用梯度裁剪防止梯度爆炸

---

## 参考文献

- TBSN: Transformer-Based Blind-Spot Network for Self-Supervised Image Denoising
- 相关论文和代码实现

---

## 许可证

请参考原始项目的许可证。

---

## 贡献

欢迎提交 Issue 和 Pull Request！

---

## 更新日志

### v1.0.0
- ✅ 提取所有核心即插即用模块
- ✅ 完成所有模块的测试
- ✅ 编写完整的使用文档

---

**如有问题，请查看代码注释或提交 Issue。**

