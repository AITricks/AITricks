# CTO架构即插即用模块说明文档

## 📋 目录

- [概述](#概述)
- [模块列表](#模块列表)
- [快速开始](#快速开始)
- [模块详细说明](#模块详细说明)
  - [Res2Net Bottle2neck](#1-res2net-bottle2neck)
  - [Stitch Attention](#2-stitch-attention)
  - [Position Attention Module](#3-position-attention-module)
  - [Channel Attention Module](#4-channel-attention-module)
  - [Dual Attention Head](#5-dual-attention-head)
  - [Sobel边界检测算子](#6-sobel边界检测算子)
  - [Boundary Enhancement Module](#7-boundary-enhancement-module)
  - [Boundary Injection Module](#8-boundary-injection-module)
- [使用示例](#使用示例)
- [测试](#测试)
- [注意事项](#注意事项)
- [参考文献](#参考文献)

---

## 概述

本文件包含从CTO (Convolution, Transformer, and Operator) 架构中提取的即插即用模块。这些模块可以独立使用，也可以组合使用，适用于各种深度学习任务，特别是医学图像分割任务。

### 主要特点

- ✅ **即插即用**：所有模块都可以独立使用，无需依赖完整的CTO架构
- ✅ **模块化设计**：每个模块功能单一，接口清晰
- ✅ **详细文档**：每个模块都有详细的说明和使用示例
- ✅ **完整测试**：所有模块都包含测试函数，确保正确性

---

## 模块列表

| 模块名称 | 功能描述 | 对应论文图 |
|---------|---------|-----------|
| Res2Net Bottle2neck | 多尺度特征提取 | Fig. 3 |
| Stitch Attention | 多尺度注意力机制 | Fig. 4 |
| Position Attention Module | 位置注意力模块 | - |
| Channel Attention Module | 通道注意力模块 | - |
| Dual Attention Head | 双注意力头 | - |
| Sobel边界检测算子 | 边界提取 | - |
| Boundary Enhancement Module | 边界增强模块 | Fig. 2(c) |
| Boundary Injection Module | 边界注入模块 | Fig. 2(c) |

---

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy
```

### 基本使用

```python
import torch
from plug_and_play_modules import (
    Res2NetBottle2neck,
    StitchAttention,
    PositionAttentionModule,
    ChannelAttentionModule,
    DualAttentionHead,
    BoundaryEnhancementModule,
    BoundaryInjectionModule,
    get_sobel,
    run_sobel
)

# 创建模块
module = Res2NetBottle2neck(inplanes=256, planes=64, baseWidth=26, scale=4)

# 前向传播
x = torch.randn(2, 256, 64, 64)
out = module(x)
print(f"输出形状: {out.shape}")
```

### 运行测试

```bash
python plug_and_play_modules.py
```

---

## 模块详细说明

### 1. Res2Net Bottle2neck

**功能**：多尺度特征提取模块，通过层次化的残差连接实现多尺度特征表示。

**对应结构图**：Fig. 3 - Basic module of Res2Net

**特点**：
- 多尺度特征提取（scale参数控制尺度数量）
- 层次化的残差连接
- 可配置的基础宽度和尺度

**使用方法**：

```python
from plug_and_play_modules import Res2NetBottle2neck

# 创建模块
module = Res2NetBottle2neck(
    inplanes=256,      # 输入通道数
    planes=64,          # 输出通道数
    baseWidth=26,       # 基础宽度
    scale=4,           # 尺度数量（对应X1, X2, X3, X4）
    stride=1,          # 卷积步长
    stype='normal'     # 'normal' 或 'stage'
)

# 前向传播
x = torch.randn(2, 256, 64, 64)
out = module(x)  # [2, 256, 64, 64]
```

**参数说明**：
- `inplanes`: 输入通道数
- `planes`: 输出通道数（实际输出为 `planes * expansion`，expansion=4）
- `baseWidth`: 基础宽度，控制内部通道数
- `scale`: 尺度数量，通常为4（对应X1, X2, X3, X4四个分支）
- `stride`: 卷积步长
- `stype`: 'normal' 或 'stage'，用于控制第一个block的行为

---

### 2. Stitch Attention

**功能**：多尺度注意力机制，通过不同的stitch rate（采样步长）实现多尺度特征采样和注意力计算。

**对应结构图**：Fig. 4 - Stitch-ViT

**特点**：
- 多尺度采样（stitch rate）
- 多头注意力机制
- 可配置的stride参数

**使用方法**：

```python
from plug_and_play_modules import StitchAttention

# 创建模块
module = StitchAttention(
    stride=[(2, 2), (4, 4), (8, 8)],  # 采样步长列表
    d_model=256                        # 特征维度
)

# 前向传播
x = torch.randn(2, 256, 64, 64)  # 注意：尺寸需要能被所有stride整除
out = module(x)  # [2, 256, 64, 64]
```

**参数说明**：
- `stride`: 采样步长列表，例如 `[(2,2), (4,4), (8,8)]`
  - 每个元组 `(ws, hs)` 表示宽度和高度方向的采样步长
  - **重要**：输入特征图的尺寸必须能被所有stride整除
- `d_model`: 输入特征维度

**注意事项**：
- 输入特征图的H和W必须能被所有stride中的ws和hs整除
- 例如：如果stride=[(2,2), (4,4), (8,8)]，则输入尺寸必须是8的倍数

---

### 3. Position Attention Module

**功能**：位置注意力模块，捕获空间位置间的依赖关系。

**特点**：
- 空间位置间的注意力计算
- 自适应权重学习

**使用方法**：

```python
from plug_and_play_modules import PositionAttentionModule

# 创建模块
module = PositionAttentionModule(in_channels=256)

# 前向传播
x = torch.randn(2, 256, 64, 64)
out = module(x)  # [2, 256, 64, 64]
```

**参数说明**：
- `in_channels`: 输入通道数

---

### 4. Channel Attention Module

**功能**：通道注意力模块，捕获通道间的依赖关系。

**特点**：
- 通道间的注意力计算
- 自适应权重学习

**使用方法**：

```python
from plug_and_play_modules import ChannelAttentionModule

# 创建模块
module = ChannelAttentionModule()

# 前向传播
x = torch.randn(2, 256, 64, 64)
out = module(x)  # [2, 256, 64, 64]
```

**注意**：该模块不需要指定输入通道数，会自动从输入特征中获取。

---

### 5. Dual Attention Head

**功能**：双注意力头模块，结合位置注意力和通道注意力。

**特点**：
- 同时使用位置注意力和通道注意力
- 可选的辅助输出

**使用方法**：

```python
from plug_and_play_modules import DualAttentionHead

# 创建模块
module = DualAttentionHead(
    in_channels=256,  # 输入通道数
    nclass=1,          # 输出类别数
    aux=False         # 是否输出辅助结果
)

# 前向传播
x = torch.randn(2, 256, 64, 64)
outputs = module(x)  # 如果aux=False，返回tuple包含1个元素
                     # 如果aux=True，返回tuple包含3个元素
```

**参数说明**：
- `in_channels`: 输入通道数
- `nclass`: 输出类别数
- `aux`: 是否输出辅助结果（位置注意力和通道注意力的单独输出）

**返回值**：
- 如果 `aux=False`：返回 `(fusion_out,)`
- 如果 `aux=True`：返回 `(fusion_out, p_out, c_out)`

---

### 6. Sobel边界检测算子

**功能**：使用Sobel算子提取图像边界信息。

**特点**：
- 可学习的边界检测
- 支持多通道输入

**使用方法**：

```python
from plug_and_play_modules import get_sobel, run_sobel

# 创建Sobel算子
sobel_x, sobel_y = get_sobel(in_chan=3, out_chan=1)

# 运行Sobel算子
x = torch.randn(2, 3, 256, 256)
out = run_sobel(sobel_x, sobel_y, x)  # [2, 3, 256, 256]
```

**参数说明**：
- `in_chan`: 输入通道数
- `out_chan`: 输出通道数（通常为1）

**注意**：Sobel算子的权重是固定的，不参与训练。

---

### 7. Boundary Enhancement Module

**功能**：边界增强模块，融合多尺度边界信息。

**对应结构图**：Fig. 2(c) - Boundary Enhancement Module (BEM)

**特点**：
- 融合深层和浅层特征
- 输出边界特征图

**使用方法**：

```python
from plug_and_play_modules import BoundaryEnhancementModule

# 创建模块
module = BoundaryEnhancementModule()

# 前向传播
x1 = torch.randn(2, 256, 64, 64)   # 浅层特征
x4 = torch.randn(2, 2048, 8, 8)    # 深层特征
out = module(x4, x1)  # [2, 1, 64, 64]
```

**参数说明**：
- `x4`: 深层特征 [B, 2048, H/8, W/8]
- `x1`: 浅层特征 [B, 256, H/4, W/4]

**返回值**：边界特征图 [B, 1, H/4, W/4]

---

### 8. Boundary Injection Module

**功能**：边界注入模块，将边界信息注入到解码器特征中。

**对应结构图**：Fig. 2(c) - Boundary Injection Module (BIM)

**特点**：
- 将边界信息注入解码器
- 多路径特征融合

**使用方法**：

```python
from plug_and_play_modules import BoundaryInjectionModule

# 创建模块
module = BoundaryInjectionModule()

# 前向传播
xr = torch.randn(2, 64, 64, 64)           # 解码器特征
dualattention = torch.randn(2, 64, 32, 32)  # 双注意力特征
out = module(xr, dualattention)  # [2, 1, 64, 64]
```

**参数说明**：
- `xr`: 解码器特征 [B, 64, H, W]
- `dualattention`: 双注意力特征 [B, 64, H', W']

**返回值**：注入边界信息后的特征 [B, 1, H, W]

---

## 使用示例

### 示例1：构建一个简单的边界增强网络

```python
import torch
import torch.nn as nn
from plug_and_play_modules import (
    BoundaryEnhancementModule,
    BoundaryInjectionModule,
    get_sobel,
    run_sobel
)

class SimpleBoundaryNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.bem = BoundaryEnhancementModule()
        self.bim = BoundaryInjectionModule()
        self.sobel_x, self.sobel_y = get_sobel(256, 1)
        
    def forward(self, x1, x4, decoder_feat):
        # 提取边界
        s1 = run_sobel(self.sobel_x, self.sobel_y, x1)
        s4 = run_sobel(self.sobel_x, self.sobel_y, x4)
        
        # 边界增强
        boundary = self.bem(s4, s1)
        
        # 边界注入
        output = self.bim(decoder_feat, boundary)
        
        return output

# 使用
model = SimpleBoundaryNetwork()
x1 = torch.randn(2, 256, 64, 64)
x4 = torch.randn(2, 2048, 8, 8)
decoder_feat = torch.randn(2, 64, 64, 64)
out = model(x1, x4, decoder_feat)
```

### 示例2：使用注意力模块增强特征

```python
import torch
import torch.nn as nn
from plug_and_play_modules import (
    PositionAttentionModule,
    ChannelAttentionModule,
    DualAttentionHead
)

class AttentionEnhancedNetwork(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.pam = PositionAttentionModule(in_channels)
        self.cam = ChannelAttentionModule()
        self.dual_head = DualAttentionHead(in_channels, nclass=1, aux=False)
        
    def forward(self, x):
        # 位置注意力
        pam_out = self.pam(x)
        
        # 通道注意力
        cam_out = self.cam(x)
        
        # 双注意力融合
        dual_out = self.dual_head(x)
        
        # 特征融合
        enhanced = pam_out + cam_out + dual_out[0]
        
        return enhanced

# 使用
model = AttentionEnhancedNetwork()
x = torch.randn(2, 256, 64, 64)
out = model(x)
```

### 示例3：多尺度特征提取

```python
import torch
import torch.nn as nn
from plug_and_play_modules import (
    Res2NetBottle2neck,
    StitchAttention
)

class MultiScaleNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.res2net = Res2NetBottle2neck(
            inplanes=256, planes=64, baseWidth=26, scale=4
        )
        self.stitch_attn = StitchAttention(
            stride=[(2, 2), (4, 4), (8, 8)],
            d_model=256
        )
        
    def forward(self, x):
        # Res2Net多尺度特征
        res2net_out = self.res2net(x)
        
        # Stitch Attention多尺度注意力
        stitch_out = self.stitch_attn(x)
        
        # 特征融合
        fused = res2net_out + stitch_out
        
        return fused

# 使用
model = MultiScaleNetwork()
x = torch.randn(2, 256, 64, 64)  # 注意：64能被2, 4, 8整除
out = model(x)
```

---

## 测试

运行完整测试：

```bash
python plug_and_play_modules.py
```

测试输出示例：

```
============================================================
CTO架构即插即用模块测试
============================================================

==================================================
测试 Res2Net Bottle2neck 模块
==================================================
输入形状: torch.Size([2, 256, 64, 64])
输出形状: torch.Size([2, 256, 64, 64])
✓ Res2Net Bottle2neck 测试通过

... (其他模块测试)

============================================================
✓ 所有模块测试通过！
============================================================
```

---

## 注意事项

### 1. Stitch Attention的尺寸要求

**重要**：使用 `StitchAttention` 时，输入特征图的尺寸必须能被所有stride整除。

```python
# ✅ 正确：64能被2, 4, 8整除
x = torch.randn(2, 256, 64, 64)
module = StitchAttention(stride=[(2,2), (4,4), (8,8)], d_model=256)

# ❌ 错误：64不能被3整除
x = torch.randn(2, 256, 64, 64)
module = StitchAttention(stride=[(2,2), (3,3), (4,4)], d_model=256)
```

### 2. 通道数匹配

使用模块组合时，注意通道数的匹配：

```python
# 确保通道数匹配
x1 = torch.randn(2, 256, 64, 64)   # 256通道
x4 = torch.randn(2, 2048, 8, 8)    # 2048通道
bem = BoundaryEnhancementModule()  # 内部处理通道数转换
out = bem(x4, x1)
```

### 3. 设备选择

所有模块都支持CPU和GPU：

```python
# CPU
module = Res2NetBottle2neck(inplanes=256, planes=64)
x = torch.randn(2, 256, 64, 64)
out = module(x)

# GPU
device = torch.device('cuda')
module = module.to(device)
x = x.to(device)
out = module(x)
```

### 4. 训练模式

模块默认处于训练模式，推理时需要设置为评估模式：

```python
module.eval()
with torch.no_grad():
    out = module(x)
```

---

## 参考文献

1. **CTO论文**：
   - Lin, Y., Zhang, D., Fang, X., Chen, Y., Cheng, K. T., & Chen, H. (2025). Rethinking boundary detection in deep learning-based medical image segmentation. *Medical Image Analysis*.

2. **Res2Net论文**：
   - Gao, S. H., Cheng, M. M., Zhao, K., Zhang, X. Y., Yang, M. H., & Torr, P. H. (2019). Res2Net: A new multi-scale backbone architecture. *IEEE transactions on pattern analysis and machine intelligence*.

3. **相关代码**：
   - CTO官方代码：https://github.com/xiaofang007/CTO

---

## 许可证

本代码遵循原CTO项目的许可证。

---

## 更新日志

- **2025-01-XX**: 初始版本，提取8个即插即用模块
  - Res2Net Bottle2neck
  - Stitch Attention
  - Position/Channel Attention Modules
  - Dual Attention Head
  - Sobel边界检测算子
  - Boundary Enhancement/Injection Modules

---

## 贡献

欢迎提交Issue和Pull Request！

---

## 联系方式

如有问题，请参考原CTO项目的Issue页面。

