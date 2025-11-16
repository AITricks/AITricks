# SCConv: Spatial and Channel Reconstruction Convolution

## 简介

SCConv（Spatial and Channel reconstruction Convolution）是一种即插即用的卷积模块，用于提升卷积神经网络的特征表达能力。该模块通过**空间重构单元（SRU）**和**通道重构单元（CRU）**来分别优化特征图的空间和通道信息。

## 架构概览

SCConv模块的整体架构如图1所示，集成在ResBlock中：

```
Previous ConvBlock → 1x1 Conv → [SRU → CRU] → 1x1 Conv → Next ConvBlock
                           ↑                                    ↓
                           └────────── Skip Connection ─────────┘
```

### 主要组件

1. **SRU (Spatial Reconstruction Unit)**: 空间重构单元
2. **CRU (Channel Reconstruction Unit)**: 通道重构单元
3. **SCConv**: 将SRU和CRU串联的完整模块

## 模块详解

### 1. Spatial Reconstruction Unit (SRU)

SRU负责对特征图进行空间重构，通过门控机制分离信息丰富和信息较少的区域，然后进行交叉重构。

#### 架构流程（图2）

```
Input Feature X
    ↓
Group Normalization (GN)
    ↓
计算权重 w_i = γ_i / Σ_j γ_j
    ↓
Sigmoid(gn_x * w_gamma)
    ↓
Threshold → 分离为 W1 和 W2
    ↓
X * W1 → X1^W    X * W2 → X2^W
    ↓                ↓
X11^W + X22^W    X12^W + X21^W
    ↓                ↓
    └─── Concatenate ────→ Spatial-Refined Feature X^W
```

#### 关键步骤

1. **Group Normalization**: 
   - 使用可训练的γ参数衡量每个通道的空间信息丰富程度
   - 空间信息越丰富，γ值越大

2. **权重计算**:
   - `w_i = γ_i / Σ_j γ_j` - 归一化gamma值
   - `reweights = Sigmoid(gn_x * w_gamma)` - 计算重加权值

3. **门控机制**:
   - 通过阈值（默认0.5）分离信息丰富和信息较少的特征图
   - `info_mask = reweights >= threshold`
   - `noninfo_mask = reweights < threshold`

4. **交叉重构**:
   - 将两部分特征图各分成两半
   - 交叉相加：`[x_11 + x_22, x_12 + x_21]`
   - 能够更有效地联合两个特征并加强特征之间的交互

### 2. Channel Reconstruction Unit (CRU)

CRU负责对特征图进行通道重构，通过分离、变换和融合三个阶段来优化通道信息。

#### 架构流程（图3）

```
Spatial-Refined Feature X^W (C channels)
    ↓
    ├─→ 1x1 Conv → Xup (αC channels)
    │       ↓
    │   1x1 Conv (Squeeze) → up
    │       ↓
    │   ├─→ GWC (Group-wise Conv) ─┐
    │   └─→ PWC (Point-wise Conv) ─┼→ + → Y1
    │
    └─→ 1x1 Conv → Xlow ((1-α)C channels)
            ↓
        1x1 Conv (Squeeze) → low
            ↓
        ├─→ PWC → ─┐
        └─→ low ───┼→ Concat → Y2
                    ↓
    ┌───────────────────────────┐
    │ Fuse Stage:               │
    │ Y1 → Pooling → S1 ─┐     │
    │ Y2 → Pooling → S2 ─┼→ Concat │
    │                     ↓     │
    │                  SoftMax  │
    │                     ↓     │
    │                   β1, β2  │
    │                     ↓     │
    │            Y1*β1 + Y2*β2  │
    └───────────────────────────┘
    ↓
Channel-Refined Feature Y
```

#### 三个阶段

##### 1. Split（分割）

- 使用1x1卷积将输入特征图分割成两部分：
  - **Xup**: αC 通道（上半部分）
  - **Xlow**: (1-α)C 通道（下半部分）
- 默认α = 0.5，即各占一半

##### 2. Transform（变换）

**Upper Branch (上半部分)**:
- 先通过1x1卷积压缩通道（squeeze_radio = 2）
- 然后并行应用：
  - **GWC (Group-wise Convolution)**: 组卷积，捕获局部特征
  - **PWC (Point-wise Convolution)**: 点卷积，捕获全局特征
- 最后求和：`Y1 = GWC(up) + PWC1(up)`

**Lower Branch (下半部分)**:
- 先通过1x1卷积压缩通道
- 然后：
  - 一部分通过PWC变换
  - 另一部分保持不变
- 最后拼接：`Y2 = Concat([PWC2(low), low])`

##### 3. Fuse（融合）

- 分别对Y1和Y2做全局平均池化得到S1和S2
- Concatenate S1和S2
- 应用SoftMax得到通道级别的attention权重β1和β2
- 加权融合：`Y = Y1 * β1 + Y2 * β2`

## 使用方法

### 基本用法

```python
import torch
from scconv_correct import SCConv

# 创建SCConv模块
# op_channel: 输入/输出通道数（必须与输入特征图的通道数一致）
model = SCConv(op_channel=64)

# 前向传播
input_tensor = torch.randn(2, 64, 128, 128)  # [batch, channels, height, width]
output = model(input_tensor)
print(f"输入形状: {input_tensor.shape}")
print(f"输出形状: {output.shape}")  # 输出形状与输入相同
```

### 参数说明

#### SCConv 参数

- `op_channel` (int, **必需**): 操作通道数量，必须等于输入特征图的通道数
- `group_num` (int, default=16): Group Normalization的组数
- `gate_threshold` (float, default=0.5): SRU中用于分离信息的阈值
- `alpha` (float, default=0.5): CRU中分割比例，范围(0, 1)
- `squeeze_radio` (int, default=2): CRU中的压缩率
- `group_size` (int, default=2): CRU中组卷积的组大小
- `group_kernel_size` (int, default=3): CRU中组卷积的核大小

### 在ResNet中集成

```python
import torch.nn as nn
from scconv_correct import SCConv

class ResBlockWithSCConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.scconv = SCConv(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1, 1, 0, bias=False)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.scconv(out)
        out = self.conv2(out)
        out = out + residual  # Skip connection
        return out
```

## 代码结构

```
scconv_correct.py
├── GroupBatchnorm2d: 二维组归一化层
├── SRU: 空间重构单元
│   ├── __init__: 初始化GN、阈值、Sigmoid
│   ├── forward: 前向传播（GN → 权重计算 → 门控 → 重构）
│   └── reconstruct: 交叉重构方法
├── CRU: 通道重构单元
│   ├── __init__: 初始化Split、Transform、Fuse的各层
│   └── forward: 前向传播（Split → Transform → Fuse）
└── SCConv: 完整的SCConv模块
    ├── __init__: 初始化SRU和CRU
    └── forward: 串联SRU和CRU
```

## 关键设计要点

### 1. 空间重构（SRU）

- **核心思想**: 通过学习到的gamma参数识别空间信息丰富的区域
- **门控机制**: 使用阈值分离信息，避免冗余计算
- **交叉重构**: 通过交叉相加加强特征交互，提升表达能力

### 2. 通道重构（CRU）

- **非对称设计**: Upper和Lower分支采用不同的变换策略
  - Upper: GWC + PWC，捕获不同尺度的特征
  - Lower: PWC + 恒等映射，保持信息流
- **自适应融合**: 使用attention机制动态平衡两个分支的贡献

### 3. 即插即用

- 输入输出形状完全一致：`[B, C, H, W] → [B, C, H, W]`
- 可以无缝集成到现有的CNN架构中
- 不需要改变网络的其他部分

## 注意事项

1. **通道数匹配**: `op_channel`参数必须等于输入特征图的通道数
2. **内存消耗**: SCConv包含多个卷积层，会增加一定的计算量和内存消耗
3. **训练稳定性**: 建议在训练初期使用较小的学习率，让模块充分学习
4. **组数设置**: `group_num`应能被通道数整除，否则可能导致维度不匹配

## 性能特点

- ✅ **即插即用**: 无需修改网络结构，可直接替换标准卷积
- ✅ **特征增强**: 通过空间和通道重构提升特征表达能力
- ✅ **自适应学习**: 使用attention机制自适应融合特征
- ✅ **计算高效**: 相比一些复杂的注意力机制，计算开销相对较小

## 参考论文

本实现基于SCConv论文的架构设计，完整实现了：
- Figure 1: SCConv在ResBlock中的集成架构
- Figure 2: Spatial Reconstruction Unit (SRU)的详细架构
- Figure 3: Channel Reconstruction Unit (CRU)的详细架构

## 文件说明

- `scconv_correct.py`: SCConv模块的完整实现
- `plug_modules.py`: 参考实现（可能存在问题）
- `README.md`: 本说明文档

## 测试

运行以下命令测试模块：

```python
python scconv_correct.py
```

预期输出：
```
==================================================
测试 SCConv 模块
==================================================
输入形状: torch.Size([2, 64, 128, 128])
输出形状: torch.Size([2, 64, 128, 128])

✓ 所有测试通过！模块工作正常。
==================================================
```

## 许可证

本项目仅供学习和研究使用。
