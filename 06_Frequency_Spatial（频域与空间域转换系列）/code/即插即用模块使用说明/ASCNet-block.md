# ASCNet 可插拔模块架构说明文档

本文档详细说明 `model/plug_modules.py` 中的可插拔模块与论文 Fig. 3 的对应关系。

## 概述

`plug_modules.py` 实现了 ASCNet 论文 Fig. 3 中展示的三个核心模块，这些模块可以从完整的 ASCNet 网络中独立使用，便于在其他网络中复用。

## 模块与 Fig. 3 的对应关系

### 1. Pixel Shuffle (PS) 模块

**对应代码**: `PixelShuffleUpsample`  
**对应图片**: Fig. 3 (a) - 蓝色上采样箭头

#### 图片描述
在 Fig. 3 (a) 中，完整的 PS 模块流程为：
```
输入 (C x H/2 x W/2) 
  -> 1x1 Conv 
  -> CNCM 
  -> 3x3 Conv 
  -> Pixel Shuffle 
  -> 输出 (C/2 x H x W)
```

#### 代码实现
`PixelShuffleUpsample` 实现了 Pixel Shuffle 操作的核心部分。在实际使用中，可以与其他模块组合：

```python
# 完整使用示例（对应 Fig. 3 (a)）
x = conv1x1(x)           # 1x1 Conv
x = cncm(x)               # CNCM (NewBlock)
x = conv3x3(x)           # 3x3 Conv
x = PixelShuffleUpsample(upscale_factor=2)(x)  # Pixel Shuffle
```

#### 输入输出
- **输入**: `(B, C*r², H, W)`，其中 `r = upscale_factor`
- **输出**: `(B, C, H*r, W*r)`
- **示例**: 输入 `(1, 128, 32, 32)`，`upscale_factor=2` → 输出 `(1, 32, 64, 64)`

---

### 2. Residual Haar Discrete Wavelet Transform (RHDWT) 模块

**对应代码**: `RHDWTBlock`  
**对应图片**: Fig. 3 (b) - 红色下采样箭头

#### 图片描述
在 Fig. 3 (b) 中，RHDWT 模块的完整流程为：

1. **小波分解**: 输入 `Ii (C x H x W)` 通过 HDWT 分解为 4 个子带：
   - 低频子带 `Ill (C x H/2 x W/2)`
   - 高频子带 `Ilh, Ihl, Ihh`，每个 `(C x H/2 x W/2)`

2. **通道拼接**: 将 4 个子带拼接为 `(4C x H/2 x W/2)`

3. **主干分支**:
   ```
   拼接特征 (4C x H/2 x W/2)
     -> 3x3 Conv 
     -> LeakyReLU 
     -> 3x3 Conv 
     -> I_out_model (ηC x H/2 x W/2)
   ```

4. **残差分支**:
   ```
   输入 Ii (C x H x W)
     -> 3x3 Conv (stride=2)
     -> I_out_res (ηC x H/2 x W/2)
   ```

5. **残差连接**: `IR = I_out_model + I_out_res`

#### 代码实现
`RHDWTBlock` 完整实现了上述流程：

```python
class RHDWTBlock(nn.Module):
    def forward(self, x):
        # 1. 小波分解（如果可用）
        if self.use_wavelet:
            yl, yh = self.dwt(x)  # yl: 低频, yh: 高频
            feat = self._transform(yl, yh)  # 拼接为 4C x H/2 x W/2
        else:
            feat = x  # 退化实现
        
        # 2. 主干分支
        out = self.mapper(feat)  # 3x3 Conv -> LeakyReLU
        
        # 3. 残差分支
        res = self.identity(x)  # 3x3 Conv (stride=2)
        
        # 4. 残差连接
        return out + res
```

#### 输入输出
- **输入**: `(B, in_channels, H, W)`
- **输出**: `(B, out_channels, H/2, W/2)` - 空间尺寸减半
- **示例**: 输入 `(1, 32, 64, 64)`，`out_channels=64` → 输出 `(1, 64, 32, 32)`

#### 注意事项
- 如果安装了 `pytorch_wavelets`，使用真实的 Haar 小波变换
- 如果未安装，退化为步长卷积的近似实现（功能可运行，但精度略有差异）

---

### 3. Column Non-uniformity Correction Module (CNCM)

**对应代码**: `NewBlock`  
**对应图片**: Fig. 3 (c) - 橙色矩形模块

#### 图片描述
在 Fig. 3 (c) 中，CNCM 模块的完整结构为：

```
输入 (2C x H/4 x W/4)
  -> 3x3 Conv
  -> RCSSC_1 (处理 channel_in/2 通道)
  -> 级联 [原始输入, RCSSC_1 输出]
  -> 3x3 Conv
  -> RCSSC_2 (处理 channel_in/2 通道)
  -> 级联 [RCSSC_2 输出, 上一步级联结果]
  -> 1x1 Conv (压缩)
  -> 3x3 Conv
  -> 残差连接 [最终输出 + 原始输入]
  -> 输出 (2C x H/4 x W/4)
```

#### 代码实现
`NewBlock` 完整实现了上述流程：

```python
class NewBlock(nn.Module):
    def forward(self, x):
        residual = x
        
        # 第一个 RCSSC 分支
        c1 = self.unit_1(self.conv1(x))  # 通道减半 -> RCSSC_1
        
        # 级联
        x = torch.cat([residual, c1], 1)  # channel_in + channel_in/2
        
        # 第二个 RCSSC 分支
        c2 = self.unit_2(self.conv2(x))  # 压缩 -> RCSSC_2
        
        # 级联
        x = torch.cat([c2, x], 1)  # channel_in/2 + 3*channel_in/2
        
        # 压缩和融合
        x = self.conv3(x)  # 1x1 Conv (压缩) -> 3x3 Conv (融合)
        
        # 残差连接
        return x + residual
```

#### 输入输出
- **输入**: `(B, channel_in, H, W)`
- **输出**: `(B, channel_in, H, W)` - 尺寸保持不变
- **示例**: 输入 `(1, 32, 64, 64)` → 输出 `(1, 32, 64, 64)`

---

### 4. Residual Column Spatial Self-Correction (RCSSC)

**对应代码**: `RCSSC`  
**对应图片**: Fig. 3 (c) CNCM 模块内部的核心组件

#### 图片描述
RCSSC 是 CNCM 模块内部的核心组件，由三个分支组成：

1. **空间注意力分支 (SA)**:
   - 通道池化（最大池化 + 平均池化）
   - 3x3 卷积生成空间注意力图
   - Sigmoid 激活生成注意力权重

2. **通道注意力分支 (CA)**:
   - 列方向自适应池化（保留宽度维度）
   - 分别对高度和宽度维度计算注意力
   - 生成通道注意力权重

3. **低频上下文分支 (SC)**:
   - 平均池化下采样（提取低频信息）
   - 3x3 卷积处理
   - 上采样回原尺寸
   - 与原始特征相加并激活

#### 代码实现
`RCSSC` 完整实现了三个分支的融合：

```python
class RCSSC(nn.Module):
    def forward(self, x):
        res = x
        
        # 头部特征提取
        x = self.head(x)  # 3x3 Conv -> LeakyReLU
        
        # 空间注意力分支
        sa_branch = self.SA(x)
        
        # 通道注意力分支
        ca_branch = self.CA(x)
        
        # 融合空间和通道注意力
        x1 = torch.cat([sa_branch, ca_branch], dim=1)
        x1 = self.conv1x1(x1)  # 1x1 Conv -> 3x3 Conv
        
        # 低频上下文分支
        sc_out = F.interpolate(self.SC(x), x.size()[2:])
        x2 = torch.sigmoid(x + sc_out)
        
        # 两个分支相乘融合
        out = torch.mul(x1, x2)
        
        # 尾部卷积 + 残差连接
        out = self.tail(out)  # 3x3 Conv
        out = out + res
        return self.ReLU(out)
```

#### 输入输出
- **输入**: `(B, n_feat, H, W)`
- **输出**: `(B, n_feat, H, W)` - 尺寸保持不变
- **示例**: 输入 `(1, 32, 64, 64)` → 输出 `(1, 32, 64, 64)`

---

## 辅助模块

### ChannelPool
用于空间注意力模块中压缩通道维度，将最大池化和平均池化结果拼接。

### Basic
基础卷积块：Conv2d + BatchNorm + LeakyReLU，用于构建其他复杂模块。

### CALayer
通道注意力层，实现列方向的通道注意力机制，用于 RCSSC 模块中。

### spatial_attn_layer
空间注意力层，实现空间维度的注意力机制，用于 RCSSC 模块中。

---

## 在完整网络中的使用

在 ASCNet 的完整架构中（对应 Fig. 3 整体结构），这些模块的使用方式为：

### 编码器路径（下采样）
1. 输入特征 `F0 (C x H x W)` → 3x3 Conv + LeakyReLU → `F1 (C x H x W)`
2. `F1` → **RHDWTBlock** → `(2C x H/2 x W/2)`
3. → **NewBlock (CNCM)** → `(2C x H/2 x W/2)`
4. → **RHDWTBlock** → `(4C x H/4 x W/4)`
5. → **NewBlock (CNCM)** → `(4C x H/4 x W/4)`
6. → **RHDWTBlock** → `(8C x H/8 x W/8)`
7. → **NewBlock (CNCM)** → `(8C x H/8 x W/8)`

### 解码器路径（上采样）
1. 最深特征 `(8C x H/8 x W/8)` → 通道拼接（Skip Connection）
2. → 1x1 Conv → **NewBlock (CNCM)** → 3x3 Conv → **PixelShuffleUpsample** → `(4C x H/4 x W/4)`
3. → 通道拼接 → 1x1 Conv → **NewBlock (CNCM)** → 3x3 Conv → **PixelShuffleUpsample** → `(2C x H/2 x W/2)`
4. → 通道拼接 → 1x1 Conv → **NewBlock (CNCM)** → 3x3 Conv → **PixelShuffleUpsample** → `(C x H x W)`
5. → Tanh 激活 → 与输入相加 → 输出

---

## 使用示例

### 基本使用

```python
from model.plug_modules import RCSSC, NewBlock, RHDWTBlock, PixelShuffleUpsample
import torch

# 创建模块
rcssc = RCSSC(n_feat=32, reduction=16)
cncm = NewBlock(channel_in=64, reduction=16)
rhdwt = RHDWTBlock(in_channels=32, out_channels=64)
ps = PixelShuffleUpsample(upscale_factor=2)

# 测试输入
x = torch.randn(1, 32, 64, 64)

# RCSSC: 尺寸不变
y1 = rcssc(x)  # (1, 32, 64, 64)

# CNCM: 尺寸不变
x2 = torch.randn(1, 64, 64, 64)
y2 = cncm(x2)  # (1, 64, 64, 64)

# RHDWT: 空间尺寸减半
y3 = rhdwt(x)  # (1, 64, 32, 32)

# Pixel Shuffle: 空间尺寸扩大
x4 = torch.randn(1, 128, 32, 32)  # 需要 r² 倍通道
y4 = ps(x4)  # (1, 32, 64, 64)
```

### 组合使用（对应 Fig. 3 (a)）

```python
import torch.nn as nn
from model.plug_modules import NewBlock, PixelShuffleUpsample

class PSModule(nn.Module):
    """完整的 Pixel Shuffle 模块（对应 Fig. 3 (a)）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, 1)
        self.cncm = NewBlock(channel_in=in_channels, reduction=16)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels * 4, 3, padding=1)
        self.ps = PixelShuffleUpsample(upscale_factor=2)
    
    def forward(self, x):
        x = self.conv1x1(x)
        x = self.cncm(x)
        x = self.conv3x3(x)
        x = self.ps(x)
        return x
```

---

## 总结

| 模块 | 代码类 | Fig. 3 对应 | 主要功能 | 输入输出尺寸 |
|------|--------|-------------|----------|--------------|
| Pixel Shuffle | `PixelShuffleUpsample` | (a) 蓝色箭头 | 上采样 | `(C*r², H, W)` → `(C, H*r, W*r)` |
| RHDWT | `RHDWTBlock` | (b) 红色箭头 | 下采样（小波变换） | `(C, H, W)` → `(out_C, H/2, W/2)` |
| CNCM | `NewBlock` | (c) 橙色矩形 | 列非均匀性校正 | `(C, H, W)` → `(C, H, W)` |
| RCSSC | `RCSSC` | (c) 内部组件 | 残差列空间自校正 | `(C, H, W)` → `(C, H, W)` |

所有模块都经过精心设计，可以独立使用或组合使用，便于在不同网络架构中复用。

