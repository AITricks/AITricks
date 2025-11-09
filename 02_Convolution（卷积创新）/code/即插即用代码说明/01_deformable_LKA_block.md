# Deformable Large Kernel Attention (D-LKA) 使用说明

## 概述

`deformable_LKA.py` 提供了可变形大核注意力（Deformable Large Kernel Attention, D-LKA）模块，这是一个**即插即用**的注意力模块，可以直接集成到各种深度学习架构中，特别适用于医学图像分割任务。

## 模块对应关系

该模块对应论文结构图中的：
- **2D D-LKA Block (d)**: `deformable_LKA_Attention` 类
- **D-LKA Attention**: `deformable_LKA` 类（空间门控单元）
- **Deformable Convolution**: `DeformConv` 类

## 核心组件

### 1. `DeformConv` - 可变形卷积
基础的可变形卷积模块，通过学习偏移量实现自适应空间采样。

### 2. `deformable_LKA` - 可变形大核注意力核心
通过两个可变形卷积实现大感受野的注意力机制：
- 5x5 可变形深度卷积：捕获局部特征
- 7x7 可变形深度膨胀卷积（dilation=3）：捕获长距离依赖
- 等效感受野约 19x19

### 3. `deformable_LKA_Attention` - 完整注意力模块（**即插即用**）
完整的即插即用模块，包含：
- Conv 1x1 (proj_1) → GELU → deformable_LKA → Conv 1x1 (proj_2) → 残差连接

## 使用方法

### 基本使用

```python
import torch
from deformable_LKA import deformable_LKA_Attention

# 创建模块
# d_model: 特征维度（通道数），必须与输入特征图的通道数一致
d_model = 96
attention_module = deformable_LKA_Attention(d_model=d_model)

# 输入特征图
# 形状: (batch_size, channels, height, width)
x = torch.randn(1, 96, 56, 56)

# 前向传播
# 输出形状与输入相同: (1, 96, 56, 56)
output = attention_module(x)
```

### 在编码器-解码器架构中使用

#### 示例 1: 在 U-Net 解码器中使用

```python
import torch
import torch.nn as nn
from deformable_LKA import deformable_LKA_Attention

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # 使用 D-LKA 注意力模块
        self.attention = deformable_LKA_Attention(d_model=out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
    def forward(self, x, skip):
        x = self.up_conv(x)
        # 融合 skip connection
        x = x + skip
        # 应用 D-LKA 注意力
        x = self.attention(x)
        x = self.conv(x)
        return x
```

#### 示例 2: 在 Transformer 块中使用

```python
import torch
import torch.nn as nn
from deformable_LKA import deformable_LKA_Attention

class D_LKA_Block(nn.Module):
    """完整的 D-LKA Block，包含 LayerNorm 和 MLP"""
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # D-LKA 注意力模块
        self.attn = deformable_LKA_Attention(d_model=dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(mlp_hidden_dim, dim, 1)
        )
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        # 转换为 (B, C, H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        
        # 注意力分支
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + self.attn(x_norm)
        
        # MLP 分支
        x_norm = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = x + self.mlp(x_norm)
        
        # 转换回 (B, N, C)
        x = x.view(B, C, H * W).permute(0, 2, 1)
        return x
```

#### 示例 3: 直接替换现有注意力机制

```python
# 原代码使用普通注意力
# self.attention = SomeAttentionModule(dim)

# 替换为 D-LKA 注意力
from deformable_LKA import deformable_LKA_Attention
self.attention = deformable_LKA_Attention(d_model=dim)

# 注意：输入必须是 (B, C, H, W) 格式的 2D 特征图
# 如果输入是 (B, N, C) 格式，需要先 reshape
```

## 输入输出规范

### 输入要求
- **格式**: `torch.Tensor`
- **形状**: `(batch_size, channels, height, width)`
- **数据类型**: `float32`
- **设备**: CPU 或 CUDA

### 输出规范
- **形状**: 与输入形状完全相同 `(batch_size, channels, height, width)`
- **特性**: 输入输出空间尺寸不变，通道数不变

## 参数说明

### `deformable_LKA_Attention`
- **d_model** (int): 特征维度（通道数）
  - 必须与输入特征图的通道数一致
  - 例如：如果输入是 `(B, 96, H, W)`，则 `d_model=96`

### `deformable_LKA`
- **dim** (int): 特征维度（通道数）

### `DeformConv`
- **in_channels** (int): 输入通道数
- **groups** (int): 分组卷积的组数，通常等于 `in_channels`（深度可分离卷积）
- **kernel_size** (tuple): 卷积核大小，例如 `(5, 5)`
- **padding** (int): 填充大小
- **stride** (int): 步长
- **dilation** (int): 膨胀率

## 适用场景

1. **医学图像分割**: CT、MRI 等医学图像的器官分割
2. **编码器-解码器架构**: U-Net、SegNet 等
3. **Transformer 架构**: Vision Transformer、Swin Transformer 等
4. **需要长距离依赖的任务**: 目标检测、语义分割等

## 优势特点

1. **即插即用**: 可以直接替换现有网络中的注意力模块
2. **自适应采样**: 通过可变形卷积实现自适应特征采样
3. **大感受野**: 通过大核卷积和膨胀卷积捕获长距离依赖
4. **参数效率**: 使用深度可分离卷积，参数相对较少
5. **形状不变**: 输入输出形状完全相同，便于集成

## 注意事项

1. **通道数匹配**: `d_model` 必须与输入特征图的通道数一致
2. **输入格式**: 输入必须是 `(B, C, H, W)` 格式的 2D 特征图
3. **设备一致性**: 确保模块和输入数据在同一设备上（CPU 或 CUDA）
4. **内存占用**: 可变形卷积需要额外的内存存储偏移量，注意内存使用

## 依赖要求

```bash
torch >= 1.8.0
torchvision >= 0.9.0
```

## 完整示例

```python
import torch
import torch.nn as nn
from deformable_LKA import deformable_LKA_Attention

# 创建简单的分割网络
class SimpleSegmentationNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=9):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        
        # D-LKA 注意力模块
        self.attention = deformable_LKA_Attention(d_model=128)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1),
        )
        
    def forward(self, x):
        # 编码
        x = self.encoder(x)
        # 应用 D-LKA 注意力
        x = self.attention(x)
        # 解码
        x = self.decoder(x)
        return x

# 使用示例
if __name__ == '__main__':
    # 创建模型
    model = SimpleSegmentationNet(in_channels=3, num_classes=9)
    
    # 创建输入
    x = torch.randn(1, 3, 224, 224)
    
    # 前向传播
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (1, 9, 224, 224)
```

## 参考

- 论文: Deformable Large Kernel Attention for Medical Image Segmentation
- 结构图: 对应论文中的 "2D D-LKA Block" 和 "Architecture of the deformable LKA module"

## 许可证

请参考项目主目录的 LICENSE 文件。

