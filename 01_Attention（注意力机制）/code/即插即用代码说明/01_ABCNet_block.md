# ABCNet 即插即用注意力模块使用说明
# ABCNet Plug-and-Play Attention Modules User Guide

## 📋 目录 (Table of Contents)
- [概述 (Overview)](#概述-overview)
- [模块介绍 (Module Introduction)](#模块介绍-module-introduction)
- [快速开始 (Quick Start)](#快速开始-quick-start)
- [详细使用指南 (Detailed Usage Guide)](#详细使用指南-detailed-usage-guide)
- [API 参考 (API Reference)](#api-参考-api-reference)
- [示例代码 (Examples)](#示例代码-examples)
- [常见问题 (FAQ)](#常见问题-faq)

---

## 概述 (Overview)

`ABCNet_block.py` 提供了 ABC 网络的核心注意力模块，实现了真正的**即插即用**设计：

✅ **无需外部依赖**：仅使用 PyTorch 标准库  
✅ **自动参数推断**：无需手动指定特征图尺寸  
✅ **支持任意输入尺寸**：自动适配不同分辨率的输入  
✅ **灵活的批处理**：支持任意批次大小  

`ABCNet_block.py` provides core attention modules of ABC network with true **plug-and-play** design:

✅ **No external dependencies**: Only uses PyTorch standard library  
✅ **Auto parameter inference**: No need to manually specify feature map sizes  
✅ **Support arbitrary input sizes**: Automatically adapts to different resolutions  
✅ **Flexible batch processing**: Supports arbitrary batch sizes  

---

## 模块介绍 (Module Introduction)

### 1. BilinearAttention (BAM) - 双线性注意力模块

ABC 网络的核心创新，通过双线性相关性计算空间注意力。

**特点 (Features)**:
- 自动适配任意输入尺寸 (H, W)
- 无需预先指定特征图维度
- 轻量级设计，计算效率高

**使用场景 (Use Cases)**:
- 作为独立的注意力模块插入到现有网络中
- 与其他注意力机制组合使用
- 用于特征增强和空间关系建模

### 2. ConvAttention - 卷积注意力模块

结合普通卷积和扩张卷积的注意力机制，用于提取多尺度特征。

**特点 (Features)**:
- 融合局部和全局信息
- 通过扩张卷积捕获多尺度上下文
- 包含残差连接，训练稳定

**使用场景 (Use Cases)**:
- 编码器中的特征提取
- 多尺度特征融合
- 增强网络的特征表示能力

### 3. ConvTransformerBlock (CLFT) - 卷积线性融合Transformer

ABC 网络编码器中的核心模块，结合了卷积注意力和前馈网络。

**特点 (Features)**:
- 完整的 Transformer 结构
- 支持通道数变化（输入输出通道可以不同）
- 适合用于编码器阶段

**使用场景 (Use Cases)**:
- UNet 风格的编码器
- 特征提取和变换
- 多尺度特征处理

### 4. SimplifiedBAM - 简化版双线性注意力

轻量级双线性注意力模块，适合资源受限的场景。

**特点 (Features)**:
- 更低的计算复杂度
- 保持核心的双线性注意力机制
- 适合移动端或边缘设备

**使用场景 (Use Cases)**:
- 资源受限的场景
- 实时推理应用
- 轻量级模型设计

### 5. UCDC - U形卷积-扩张卷积模块

U形结构的卷积-扩张卷积模块，用于瓶颈层和解码器阶段。

**特点 (Features)**:
- U形结构，包含内部和外部skip connections
- 多尺度扩张卷积（dilation rates: 2, 4, 2）
- 捕获多尺度上下文信息
- 支持通道数变化

**使用场景 (Use Cases)**:
- 瓶颈层（bottleneck layer）
- 解码器阶段
- 需要多尺度特征融合的场景
- ABC网络的完整实现

---

## 快速开始 (Quick Start)

### 基础使用 (Basic Usage)

```python
import torch
from ABCNet_block import BilinearAttention, ConvAttention, ConvTransformerBlock, UCDC

# 创建输入特征图
x = torch.randn(2, 64, 32, 32)  # (batch, channels, height, width)

# 1. 使用 BilinearAttention
bam = BilinearAttention(in_dim=64)
att_out = bam(x)  # 输出形状: (2, 64, 32, 32)

# 2. 使用 ConvAttention
conv_att = ConvAttention(in_dim=64)
conv_out = conv_att(x)  # 输出形状: (2, 64, 32, 32)

# 3. 使用 ConvTransformerBlock
clft = ConvTransformerBlock(in_dim=64, out_dim=128)
clft_out = clft(x)  # 输出形状: (2, 128, 32, 32)

# 4. 使用 UCDC模块
ucdc = UCDC(in_ch=64, out_ch=128)
ucdc_out = ucdc(x)  # 输出形状: (2, 128, 32, 32)
```

### 集成到现有网络 (Integration into Existing Networks)

```python
import torch.nn as nn
from ABCNet_block import ConvTransformerBlock, UCDC

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器
        self.encoder1 = nn.Conv2d(3, 64, 3, padding=1)
        self.encoder2 = ConvTransformerBlock(64, 128)
        self.encoder3 = ConvTransformerBlock(128, 256)
        
        # 瓶颈层：使用UCDC模块
        self.bottleneck = UCDC(256, 512)
        
        # 解码器
        self.decoder = nn.Conv2d(512, 1, 1)
        
    def forward(self, x):
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.bottleneck(x)  # UCDC模块
        x = self.decoder(x)
        return x
```

---

## 详细使用指南 (Detailed Usage Guide)

### 1. BilinearAttention 使用

#### 基本用法

```python
from ABCNet_block import BilinearAttention

# 创建模块（只需指定通道数）
bam = BilinearAttention(in_dim=64, reduction_ratio=4)

# 前向传播（支持任意尺寸）
x1 = torch.randn(1, 64, 32, 32)   # 小尺寸
x2 = torch.randn(1, 64, 256, 256) # 大尺寸
x3 = torch.randn(1, 64, 64, 128)  # 非正方形

out1 = bam(x1)  # (1, 64, 32, 32)
out2 = bam(x2)  # (1, 64, 256, 256)
out3 = bam(x3)  # (1, 64, 64, 128)
```

#### 参数说明

- `in_dim` (int): 输入通道数，必需参数
- `reduction_ratio` (int): 注意力降维比例，默认 4

#### 注意事项

- 输入必须是 4D 张量：`(B, C, H, W)`
- 输出形状与输入相同：`(B, C, H, W)`
- 支持任意批次大小和空间尺寸

### 2. ConvAttention 使用

#### 基本用法

```python
from ABCNet_block import ConvAttention

# 创建模块
conv_att = ConvAttention(in_dim=64, reduction_ratio=4)

# 前向传播
x = torch.randn(2, 64, 64, 64)
out = conv_att(x)  # (2, 64, 64, 64)
```

#### 内部结构

```
输入 (x)
  ├─ Conv 分支 → q
  ├─ DConv 分支 → k
  └─ BilinearAttention → att
  
v = q + k
out = γ * (att * v) + v + x  (残差连接)
```

#### 参数说明

- `in_dim` (int): 输入通道数，必需参数
- `reduction_ratio` (int): BAM 的降维比例，默认 4

### 3. ConvTransformerBlock (CLFT) 使用

#### 基本用法

```python
from ABCNet_block import ConvTransformerBlock

# 创建模块（支持通道数变化）
clft = ConvTransformerBlock(in_dim=64, out_dim=128, reduction_ratio=4)

# 前向传播
x = torch.randn(2, 64, 32, 32)
out = clft(x)  # (2, 128, 32, 32)
```

#### 内部结构

```
输入 (x)
  └─ ConvAttention → x'
     └─ FeedForward → out (通道数变化)
```

#### 参数说明

- `in_dim` (int): 输入通道数，必需参数
- `out_dim` (int): 输出通道数，必需参数
- `reduction_ratio` (int): 注意力降维比例，默认 4

### 4. SimplifiedBAM 使用

#### 基本用法

```python
from ABCNet_block import SimplifiedBAM

# 创建模块
simple_bam = SimplifiedBAM(in_dim=64, reduction_ratio=8)

# 前向传播
x = torch.randn(2, 64, 64, 64)
out = simple_bam(x)  # (2, 64, 64, 64)
```

#### 适用场景

- 资源受限的场景
- 需要快速推理的应用
- 轻量级模型设计

### 5. UCDC 使用

#### 基本用法

```python
from ABCNet_block import UCDC

# 创建模块（支持通道数变化）
ucdc = UCDC(in_ch=64, out_ch=128)

# 前向传播
x = torch.randn(2, 64, 32, 32)
out = ucdc(x)  # (2, 128, 32, 32)
```

#### 内部结构

```
输入 (x)
  ↓
Conv (初始卷积) → x1
  ↓
D.C.(r=2) → dx1 ──┐
  ↓                │ (内部skip connection)
D.C.(r=4) → dx2   │
  ↓                │
D.C.(r=2) ← concat(dx1, dx2) → dx3
  ↓
Conv (最终卷积) ← concat(x1, dx3)
  ↓
输出 (out)
```

#### 参数说明

- `in_ch` (int): 输入通道数，必需参数
- `out_ch` (int): 输出通道数，必需参数

#### 特点

- **U形结构**: 包含内部和外部skip connections，保持信息流
- **多尺度扩张卷积**: 使用不同的dilation rates (2, 4, 2) 捕获多尺度特征
- **通道数灵活**: 支持输入输出通道数的变化
- **即插即用**: 无需预先指定输入尺寸，自动适配

#### 适用场景

- 瓶颈层（bottleneck layer）
- 解码器阶段
- 需要多尺度上下文信息的场景
- ABC网络的完整实现

#### 在ABC网络中的使用

```python
# 瓶颈层
bottleneck = UCDC(in_ch=256, out_ch=512)

# 解码器阶段
decoder_stage = UCDC(in_ch=512, out_ch=256)
```

---

## API 参考 (API Reference)

### BilinearAttention

```python
class BilinearAttention(nn.Module):
    """
    BAM (Bilinear Attention Module) - 双线性注意力模块
    
    Args:
        in_dim (int): 输入通道数
        reduction_ratio (int): 注意力降维比例，默认 4
    """
    def __init__(self, in_dim, reduction_ratio=4):
        ...
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)
        Returns:
            torch.Tensor: 注意力加权的输出 (B, C, H, W)
        """
        ...
```

### ConvAttention

```python
class ConvAttention(nn.Module):
    """
    卷积注意力模块：结合普通卷积和扩张卷积的注意力机制
    
    Args:
        in_dim (int): 输入通道数
        reduction_ratio (int): 注意力降维比例，默认 4
    """
    def __init__(self, in_dim, reduction_ratio=4):
        ...
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)
        Returns:
            torch.Tensor: 注意力加权的输出 (B, C, H, W)
        """
        ...
```

### ConvTransformerBlock

```python
class ConvTransformerBlock(nn.Module):
    """
    CLFT (Convolution Linear Fusion Transformer) - 卷积线性融合Transformer
    
    Args:
        in_dim (int): 输入通道数
        out_dim (int): 输出通道数
        reduction_ratio (int): 注意力降维比例，默认 4
    """
    def __init__(self, in_dim, out_dim, reduction_ratio=4):
        ...
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, in_dim, H, W)
        Returns:
            torch.Tensor: 变换后的特征图 (B, out_dim, H, W)
        """
        ...
```

### SimplifiedBAM

```python
class SimplifiedBAM(nn.Module):
    """
    简化版BAM - 轻量级双线性注意力模块
    
    Args:
        in_dim (int): 输入通道数
        reduction_ratio (int): 注意力降维比例，默认 8
    """
    def __init__(self, in_dim, reduction_ratio=8):
        ...
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, C, H, W)
        Returns:
            torch.Tensor: 注意力加权的输出 (B, C, H, W)
        """
        ...
```

### UCDC

```python
class UCDC(nn.Module):
    """
    UCDC (U-shaped Convolution-Dilated Convolution) - U形卷积-扩张卷积模块
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
    """
    def __init__(self, in_ch, out_ch):
        ...
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图 (B, in_ch, H, W)
        Returns:
            torch.Tensor: 输出特征图 (B, out_ch, H, W)
        """
        ...
```

---

## 示例代码 (Examples)

### 示例 1：基础使用

```python
import torch
from ABCNet_block import BilinearAttention, ConvAttention, ConvTransformerBlock, UCDC

# 创建输入
x = torch.randn(2, 64, 32, 32)
print(f"输入形状: {x.shape}")

# BAM模块
bam = BilinearAttention(in_dim=64)
bam_out = bam(x)
print(f"BAM输出形状: {bam_out.shape}")

# CLFT模块
clft = ConvTransformerBlock(in_dim=64, out_dim=128)
clft_out = clft(x)
print(f"CLFT输出形状: {clft_out.shape}")

# UCDC模块
ucdc = UCDC(in_ch=64, out_ch=128)
ucdc_out = ucdc(x)
print(f"UCDC输出形状: {ucdc_out.shape}")
```

### 示例 2：集成到UNet

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ABCNet_block import ConvTransformerBlock, ConvAttention, UCDC

class UNetWithABC(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc2 = ConvTransformerBlock(64, 128)
        self.enc3 = ConvTransformerBlock(128, 256)
        self.enc4 = ConvTransformerBlock(256, 512)
        
        # 瓶颈层：使用UCDC模块
        self.bottleneck = UCDC(512, 1024)
        
        # 解码器：使用UCDC模块
        self.dec4 = UCDC(1024, 512)
        self.dec3 = ConvTransformerBlock(512, 256)
        self.dec2 = ConvTransformerBlock(256, 128)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.final = nn.Conv2d(64, out_channels, 1)
        
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # 编码器
        e1 = F.relu(self.enc1(x))
        e2 = self.pool(e1)
        e2 = self.enc2(e2)
        e3 = self.pool(e2)
        e3 = self.enc3(e3)
        e4 = self.pool(e3)
        e4 = self.enc4(e4)
        
        # 瓶颈层：UCDC模块
        b = self.pool(e4)
        b = self.bottleneck(b)
        
        # 解码器
        d4 = self.up(b)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)  # UCDC模块
        
        d3 = self.up(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = F.relu(self.dec1(d1))
        out = self.final(d1)
        
        return out

# 测试
model = UNetWithABC()
x = torch.randn(1, 3, 256, 256)
out = model(x)
print(f"输出形状: {out.shape}")  # (1, 1, 256, 256)
```

### 示例 2.5：UCDC模块详细使用

```python
import torch
import torch.nn as nn
from ABCNet_block import UCDC

# 基础使用
ucdc = UCDC(in_ch=64, out_ch=128)
x = torch.randn(2, 64, 32, 32)
out = ucdc(x)
print(f"UCDC - 输入: {x.shape}, 输出: {out.shape}")

# 在完整网络中使用
class NetworkWithUCDC(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 256, 3, padding=1)
        # 瓶颈层：使用UCDC
        self.bottleneck = UCDC(256, 512)
        # 解码器：也使用UCDC
        self.decoder = UCDC(512, 256)
        self.output = nn.Conv2d(256, 1, 1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)  # UCDC模块
        x = self.decoder(x)      # UCDC模块
        x = self.output(x)
        return x

model = NetworkWithUCDC()
x = torch.randn(1, 3, 128, 128)
out = model(x)
print(f"完整网络输出: {out.shape}")
```

### 示例 3：多尺度特征融合

```python
import torch
import torch.nn as nn
from ABCNet_block import ConvAttention

class MultiScaleFusion(nn.Module):
    def __init__(self, in_dim=64):
        super().__init__()
        self.attention = ConvAttention(in_dim)
        self.conv = nn.Conv2d(in_dim, in_dim, 3, padding=1)
        
    def forward(self, x):
        # 应用注意力
        att_out = self.attention(x)
        # 融合
        out = self.conv(att_out + x)
        return out

# 测试
model = MultiScaleFusion(64)
x = torch.randn(2, 64, 128, 128)
out = model(x)
print(f"输出形状: {out.shape}")  # (2, 64, 128, 128)
```

### 示例 4：性能测试

```python
import torch
import time
from ABCNet_block import BilinearAttention, ConvAttention, ConvTransformerBlock, UCDC

# 测试参数
batch_size = 4
channels = 64
height, width = 64, 64

x = torch.randn(batch_size, channels, height, width)

# 测试不同模块
modules = {
    'BAM': BilinearAttention(channels),
    'ConvAttention': ConvAttention(channels),
    'CLFT': ConvTransformerBlock(channels, channels),
    'UCDC': UCDC(channels, channels),
}

for name, module in modules.items():
    # 预热
    for _ in range(10):
        _ = module(x)
    
    # 计时
    start_time = time.time()
    for _ in range(100):
        _ = module(x)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100 * 1000  # 毫秒
    print(f"{name} 平均推理时间: {avg_time:.2f} ms")
    
    # 计算参数量
    total_params = sum(p.numel() for p in module.parameters())
    print(f"{name} 参数量: {total_params:,}")
```

---

## 常见问题 (FAQ)

### Q1: 如何选择合适的模块？

**A:** 
- **BilinearAttention**: 需要轻量级空间注意力时使用
- **ConvAttention**: 需要多尺度特征融合时使用
- **ConvTransformerBlock**: 用于编码器阶段，需要改变通道数时使用
- **SimplifiedBAM**: 资源受限场景使用
- **UCDC**: 用于瓶颈层和解码器阶段，需要多尺度上下文信息时使用

### Q2: 模块是否支持可变输入尺寸？

**A:** 是的！所有模块都支持任意输入尺寸，无需预先指定。输入可以是任意 (B, C, H, W) 形状。

### Q3: 如何调整注意力强度？

**A:** 
- 调整 `reduction_ratio` 参数（较小的值 = 更强的注意力）
- 在 `ConvAttention` 中，`gamma` 参数控制注意力输出的权重（可训练）

### Q4: 模块是否支持批处理？

**A:** 是的！所有模块都支持任意批次大小。

### Q5: 如何集成到现有网络中？

**A:** 直接将模块插入到你的网络中即可，例如：
```python
class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.attention = ConvAttention(64)  # 即插即用
        self.out = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.attention(x)  # 插入注意力
        x = self.out(x)
        return x
```

### Q6: 模块的计算复杂度如何？

**A:** 
- **BilinearAttention**: O(C × H × W)，轻量级
- **ConvAttention**: O(C × H × W × K²)，K 是卷积核大小
- **ConvTransformerBlock**: O(C × H × W × K²)，包含注意力+前馈
- **SimplifiedBAM**: O(C × H × W)，最轻量级
- **UCDC**: O(C × H × W × K²)，包含多尺度扩张卷积和skip connections

### Q7: 是否需要额外的依赖？

**A:** 不需要！模块仅使用 PyTorch 标准库，无需安装其他依赖。

### Q8: 模块是否支持 GPU？

**A:** 是的！模块完全支持 GPU，只需将输入张量移动到 GPU 上：
```python
device = torch.device('cuda')
x = torch.randn(2, 64, 32, 32).to(device)
module = BilinearAttention(64).to(device)
out = module(x)
```

---

## 总结 (Summary)

`ABCNet_block.py` 提供了完整的 ABC 网络注意力模块实现，具有以下特点：

✅ **真正的即插即用**：无需配置，直接使用  
✅ **自动适配**：支持任意输入尺寸  
✅ **高效实现**：优化的计算流程  
✅ **易于集成**：可以轻松插入到现有网络  
✅ **无外部依赖**：仅使用 PyTorch 标准库  

`ABCNet_block.py` provides complete ABC network attention module implementation with:

✅ **True plug-and-play**: No configuration needed, use directly  
✅ **Auto adaptation**: Supports arbitrary input sizes  
✅ **Efficient implementation**: Optimized computation flow  
✅ **Easy integration**: Can be easily inserted into existing networks  
✅ **No external dependencies**: Only uses PyTorch standard library  

---

## 许可证 (License)

请参考项目主 LICENSE 文件。

Please refer to the main LICENSE file of the project.

---

## 联系方式 (Contact)

如有问题或建议，请提交 Issue 或 Pull Request。

For questions or suggestions, please submit an Issue or Pull Request.

