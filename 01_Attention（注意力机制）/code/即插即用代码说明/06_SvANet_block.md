# SvANet 即插即用模块使用说明

## 📋 目录

1. [概述](#概述)
2. [核心模块介绍](#核心模块介绍)
3. [模块详细说明](#模块详细说明)
4. [使用示例](#使用示例)
5. [架构对应关系](#架构对应关系)
6. [最佳实践](#最佳实践)

---

## 概述

本文档介绍 SvANet (Scale-variant Attention Network) 中的核心即插即用模块。这些模块可以独立使用，也可以组合使用，为小目标医学图像分割任务提供强大的特征提取能力。

### 核心模块列表

1. **MoCAttention** - Monte Carlo 注意力机制
2. **SqueezeExcitation** - SE 注意力机制
3. **AssembleFormer** - CNN + ViT 混合模块
4. **FGBottleneck** - 特征引导瓶颈块（MCBottleneck）
5. **LinearSelfAttention** - 线性复杂度自注意力（AssembleFormer 内部组件）

---

## 核心模块介绍

### 1. MoCAttention (Monte Carlo Attention)

**功能**: 通过 Monte Carlo 采样策略学习全局和局部特征，增强模型对不同尺度信息的感知能力。

**特点**:
- 训练时随机选择池化分辨率，增强模型泛化能力
- 支持特征顺序打乱（MoCOrder），增加随机性
- 推理时使用固定池化分辨率，保证稳定性

**对应结构图**: MCAttn 模块

### 2. SqueezeExcitation (SE Attention)

**功能**: 标准的 SE 注意力机制，通过全局平均池化和通道重标定增强特征表示。

**特点**:
- 轻量级设计
- 全局信息压缩
- 通道注意力重标定

**对应结构图**: 基础注意力组件

### 3. AssembleFormer (Assembling Tensors with Vision Transformer)

**功能**: 结合 CNN 的局部特征提取能力和 ViT 的全局建模能力，实现局部和全局特征的协同。

**特点**:
- 局部分支：3×3 卷积提取局部特征
- 全局分支：将特征图转换为 patches，通过 Transformer 处理
- 融合机制：局部特征 + 全局特征，通过拼接和投影融合

**对应结构图**: AssemFormer 模块

### 4. FGBottleneck (Feature Guide Bottleneck)

**功能**: 特征引导瓶颈块，对应结构图中的 MCBottleneck。结合卷积、注意力和 Transformer 的能力。

**特点**:
- 可配置的 SE 层（MoCAttention 或 SqueezeExcitation）
- 可配置的 ViT 层（AssembleFormer）
- 残差连接和随机深度支持

**对应结构图**: MCBottleneck 模块

---

## 模块详细说明

### MoCAttention

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `InChannels` | int | 必需 | 输入通道数 |
| `HidChannels` | int | None | 隐藏层通道数，默认自动计算 |
| `SqueezeFactor` | int | 4 | 压缩因子，用于计算隐藏层通道数 |
| `PoolRes` | list | [1, 2, 3] | 池化分辨率列表，训练时随机选择 |
| `Act` | Callable | nn.ReLU | 激活函数 |
| `ScaleAct` | Callable | nn.Sigmoid | 缩放激活函数 |
| `MoCOrder` | bool | True | 是否启用特征顺序打乱 |

#### 使用示例

```python
import torch
from test_plug_and_play_modules import MoCAttention

# 创建模块
moc_attn = MoCAttention(
    InChannels=64,
    PoolRes=[1, 2, 3],
    MoCOrder=True
)

# 前向传播
x = torch.randn(2, 64, 32, 32)  # [B, C, H, W]
out = moc_attn(x)  # [B, C, H, W]
```

#### 工作原理

1. **训练阶段**:
   - 随机选择一个池化分辨率（1×1, 2×2, 或 3×3）
   - 可选地打乱特征顺序
   - 对特征图进行池化
   - 如果池化后尺寸 > 1，随机选择一个位置

2. **推理阶段**:
   - 固定使用 1×1 全局平均池化
   - 保证输出稳定性

3. **注意力生成**:
   - 通过 SE 层（两个 1×1 卷积）生成注意力图
   - 与原始特征逐元素相乘

---

### SqueezeExcitation

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `InChannels` | int | 必需 | 输入通道数 |
| `HidChannels` | int | None | 隐藏层通道数，默认自动计算 |
| `SqueezeFactor` | int | 4 | 压缩因子 |
| `Act` | Callable | nn.ReLU | 激活函数 |
| `ScaleAct` | Callable | nn.Sigmoid | 缩放激活函数 |

#### 使用示例

```python
from test_plug_and_play_modules import SqueezeExcitation

se = SqueezeExcitation(InChannels=64)
out = se(x)  # [B, C, H, W]
```

---

### AssembleFormer

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `InChannels` | int | 必需 | 输入通道数 |
| `FfnMultiplier` | float/Sequence | 2.0 | FFN 维度倍数 |
| `NumAttnBlocks` | int | 2 | Transformer 块数量 |
| `PatchRes` | int | 2 | Patch 分辨率（H×W） |
| `Dilation` | int | 1 | 卷积膨胀率 |
| `AttnDropRate` | float | 0.0 | 注意力 dropout 率 |
| `DropRate` | float | 0.0 | Dropout 率 |
| `FfnDropRate` | float | 0.0 | FFN dropout 率 |
| `SDProb` | float | 0.0 | 随机深度概率 |
| `ViTSELayer` | Module | None | ViT 中的 SE 层 |

#### 使用示例

```python
from test_plug_and_play_modules import AssembleFormer

# 基础使用
assem_former = AssembleFormer(
    InChannels=64,
    NumAttnBlocks=2,
    PatchRes=2
)

out = assem_former(x)  # [B, C, H, W]

# 高级配置
assem_former = AssembleFormer(
    InChannels=64,
    FfnMultiplier=2.0,
    NumAttnBlocks=2,
    PatchRes=2,
    Dilation=1,
    AttnDropRate=0.1,
    DropRate=0.1,
    SDProb=0.1
)
```

#### 工作原理

1. **局部特征提取**:
   ```
   [B, C, H, W] -> Conv3x3 -> SE(可选) -> Conv1x1 -> [B, C//2, H, W]
   ```

2. **Patch 转换**:
   ```
   [B, C//2, H, W] -> Unfold -> [B, C//2, P, N]
   ```
   其中 P = PatchRes × PatchRes，N 是 patch 数量

3. **全局特征处理**:
   ```
   [B, C//2, P, N] -> Transformer × NumAttnBlocks -> [B, C//2, P, N]
   ```

4. **特征融合**:
   ```
   [B, C//2, P, N] -> Fold -> [B, C//2, H, W]
   [B, C//2, H, W] + [B, C//2, H, W] -> Concat -> [B, C, H, W]
   ```

5. **残差连接**:
   ```
   Output = Input + Dropout(FusedFeatures)
   ```

---

### FGBottleneck

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `InChannels` | int | 必需 | 输入通道数 |
| `HidChannels` | int | None | 隐藏层通道数，默认自动计算 |
| `Expansion` | float | 2.0 | 扩展倍数 |
| `Stride` | int | 1 | 卷积步长 |
| `Dilation` | int | 1 | 卷积膨胀率 |
| `DropRate` | float | 0.0 | 随机深度概率 |
| `SELayer` | Module | None | SE 层（MoCAttention 或 SqueezeExcitation） |
| `ActLayer` | Callable | None | 激活层 |
| `ViTBlock` | Module | None | ViT 块（AssembleFormer） |

#### 使用示例

```python
from test_plug_and_play_modules import FGBottleneck, MoCAttention, AssembleFormer

# 基础瓶颈块
bottleneck = FGBottleneck(InChannels=64)

# 带 MoCAttention 的瓶颈块
bottleneck_moc = FGBottleneck(
    InChannels=64,
    SELayer=MoCAttention
)

# 完整 MCBottleneck（MoCAttention + AssembleFormer）
mcbottleneck = FGBottleneck(
    InChannels=64,
    SELayer=MoCAttention,
    ViTBlock=AssembleFormer,
    NumAttnBlocks=2
)

out = mcbottleneck(x)  # [B, C, H, W]
```

#### 工作原理

1. **瓶颈结构**:
   ```
   Input -> Conv1x1 -> Conv3x3 -> SE Layer -> Conv1x1 -> Output
   ```

2. **残差连接**:
   ```
   Output = Act(Input + Dropout(BottleneckOutput))
   ```

3. **ViT 处理**:
   ```
   Output = ViTBlock(Output)
   ```

---

## 使用示例

### 示例 1: 基础模块使用

```python
import torch
from test_plug_and_play_modules import (
    MoCAttention, 
    SqueezeExcitation, 
    AssembleFormer, 
    FGBottleneck
)

# 创建测试输入
x = torch.randn(2, 64, 32, 32)  # [B, C, H, W]

# 1. MoCAttention
moc_attn = MoCAttention(InChannels=64)
out1 = moc_attn(x)

# 2. SqueezeExcitation
se = SqueezeExcitation(InChannels=64)
out2 = se(x)

# 3. AssembleFormer
assem_former = AssembleFormer(InChannels=64, NumAttnBlocks=2)
out3 = assem_former(x)

# 4. FGBottleneck
bottleneck = FGBottleneck(InChannels=64)
out4 = bottleneck(x)
```

### 示例 2: 组合使用

```python
# 完整的 MCBottleneck
mcbottleneck = FGBottleneck(
    InChannels=64,
    SELayer=MoCAttention,  # 使用 MoCAttention
    ViTBlock=AssembleFormer,  # 使用 AssembleFormer
    NumAttnBlocks=2,
    Expansion=2.0
)

out = mcbottleneck(x)
```

### 示例 3: 构建简单网络

```python
import torch.nn as nn

class SimpleSvANet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1: MCBottleneck with MoCAttention
        self.stage1 = FGBottleneck(
            InChannels=64,
            SELayer=MoCAttention,
            ViTBlock=AssembleFormer,
            NumAttnBlocks=2
        )
        
        # Stage 2: MCBottleneck with MoCAttention
        self.stage2 = FGBottleneck(
            InChannels=64,
            SELayer=MoCAttention,
            ViTBlock=AssembleFormer,
            NumAttnBlocks=2
        )
        
        # Decoder: AssembleFormer
        self.decoder = AssembleFormer(
            InChannels=64,
            NumAttnBlocks=2,
            PatchRes=2
        )
        
        # Head
        self.head = nn.Conv2d(64, num_classes, 1)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.decoder(x)
        x = self.head(x)
        return x

# 使用
model = SimpleSvANet(in_channels=3, num_classes=2)
x = torch.randn(2, 3, 224, 224)
out = model(x)  # [2, 2, 224, 224]
```

---

## 架构对应关系

### SvANet 结构图对应关系

| 结构图模块 | 代码模块 | 说明 |
|-----------|---------|------|
| MCAttn | `MoCAttention` | Monte Carlo 注意力 |
| MCBottleneck | `FGBottleneck` + `MoCAttention` + `AssembleFormer` | 特征引导瓶颈块 |
| AssemFormer | `AssembleFormer` | CNN + ViT 混合模块 |
| Cross-scale Guidance | `CSLayer` (未实现) | 跨尺度引导模块 |
| SvAttn | `CSLayer` 内部 | 尺度变体注意力 |

### 模块组合建议

1. **Encoder 路径**:
   - 使用 `FGBottleneck` + `MoCAttention` + `AssembleFormer`
   - 对应结构图中的 MCBottleneck

2. **Decoder 路径**:
   - 使用 `AssembleFormer` 处理上采样特征
   - 可结合 Cross-scale Guidance（需要实现 CSLayer）

3. **注意力选择**:
   - 小目标检测：优先使用 `MoCAttention`
   - 轻量级模型：使用 `SqueezeExcitation`
   - 高性能模型：使用 `MoCAttention` + `AssembleFormer`

---

## 最佳实践

### 1. 通道数配置

- 确保通道数能被 8 整除（`make_divisible` 会自动处理）
- 建议使用 32, 64, 128, 256 等常见通道数

### 2. Patch 分辨率选择

- `PatchRes=2`: 适合小特征图（H, W < 32）
- `PatchRes=4`: 适合中等特征图（32 ≤ H, W < 64）
- `PatchRes=8`: 适合大特征图（H, W ≥ 64）

### 3. 训练技巧

- **MoCAttention**: 训练时使用 `MoCOrder=True`，推理时自动关闭
- **随机深度**: 建议 `SDProb=0.1-0.2`，提高模型泛化能力
- **Dropout**: 注意力 dropout 建议 `0.1`，FFN dropout 建议 `0.0`

### 4. 内存优化

- 减少 `NumAttnBlocks` 可以降低内存占用
- 使用 `FfnMultiplier=1.5` 而不是 `2.0` 可以减少参数
- 对于大输入，考虑使用 `Dilation > 1` 扩大感受野

### 5. 性能调优

- **小目标检测**: 
  - 使用 `MoCAttention` with `PoolRes=[1, 2, 3]`
  - 增加 `NumAttnBlocks=3-4`

- **快速推理**:
  - 使用 `SqueezeExcitation` 代替 `MoCAttention`
  - 减少 `NumAttnBlocks=1`
  - 使用 `PatchRes=4` 或更大

- **高精度模型**:
  - 使用完整的 `FGBottleneck` + `MoCAttention` + `AssembleFormer`
  - `NumAttnBlocks=2-3`
  - `FfnMultiplier=2.0-3.0`

---

## 注意事项

1. **输入尺寸**: 确保输入特征图的 H 和 W 能被 `PatchRes` 整除
2. **设备**: 所有模块支持 CPU 和 GPU
3. **训练模式**: `MoCAttention` 在训练和推理时的行为不同
4. **梯度**: 使用 `StochasticDepth` 时，某些路径可能不参与反向传播

---

## 参考

- **项目结构图**: 参见 `readme/architecture_animation.gif`
- **原始实现**: `lib/model/modules/`
- **测试代码**: `test_plug_and_play_modules.py`

---

## 更新日志

- **v1.0** (2024): 初始版本，包含核心即插即用模块
  - MoCAttention
  - SqueezeExcitation
  - AssembleFormer (完整实现)
  - FGBottleneck

---

## 许可证

请参考项目 LICENSE 文件。

