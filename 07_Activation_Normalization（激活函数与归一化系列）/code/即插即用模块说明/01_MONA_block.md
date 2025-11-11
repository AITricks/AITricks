# Mona模块使用文档

## 目录
- [简介](#简介)
- [架构说明](#架构说明)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [使用示例](#使用示例)
- [参数说明](#参数说明)
- [集成指南](#集成指南)
- [特性说明](#特性说明)
- [常见问题](#常见问题)

## 简介

**Mona (Multi-cognitive Visual Adapter)** 是一个即插即用的适配器模块，用于视觉Transformer的参数高效微调。该模块设计用于：

- ✅ **参数高效**: 只需训练少量适配器参数，冻结预训练模型
- ✅ **即插即用**: 可以轻松插入到任何Transformer块中
- ✅ **多尺度特征**: 通过多认知卷积滤波器组捕获丰富的空间上下文信息
- ✅ **稳定训练**: 通过缩放LayerNorm和跳跃连接确保训练稳定性

## 架构说明

### Mona Layer 结构

Mona Layer 包含以下主要组件：

1. **缩放LayerNorm (Scaled LayerNorm)**
   - 使用两个可训练的缩放因子 (S1: gamma, S2: gammax)
   - 平衡归一化特征和原始特征的权重
   - 初始时gamma很小，确保适配器影响较小，训练过程中逐渐增大

2. **下投影 (Down Projection)**
   - 将特征维度从 `in_dim` 降低到 `bottleneck_dim`
   - 减少参数量，提高参数效率

3. **多认知卷积滤波器组 (MonaOp)**
   - **核心组件**: 三个并行的深度卷积 (3x3, 5x5, 7x7)
   - 捕获不同尺度的空间特征（局部、中等范围、大范围）
   - 平均聚合融合多尺度特征
   - 1x1投影层进行特征变换
   - 四个跳跃连接增强信息流

4. **GeLU激活函数**
   - 引入非线性，增强模型表达能力

5. **上投影 (Up Projection)**
   - 将特征维度从 `bottleneck_dim` 恢复到 `in_dim`
   - 保持输入输出维度一致

6. **跳跃连接 (Skip Connection)**
   - 残差连接确保信息流畅通
   - 允许适配器学习增量调整

### 架构图

```
输入 (B, N, C)
    │
    ├─→ [跳跃连接] ──────────────┐
    │                            │
    ↓                            │
[缩放LayerNorm]                  │
norm(x) * S1 + x * S2           │
    │                            │
    ↓                            │
[下投影]                         │
Linear(in_dim → bottleneck_dim)  │
    │                            │
    ↓                            │
[重塑为空间格式]                  │
(B, N, C) → (B, C, H, W)        │
    │                            │
    ↓                            │
[多认知卷积滤波器组]              │
  ├─ 3x3深度卷积                 │
  ├─ 5x5深度卷积                 │
  ├─ 7x7深度卷积                 │
  └─ 平均聚合 + 1x1投影          │
    │                            │
    ↓                            │
[重塑回序列格式]                  │
(B, C, H, W) → (B, N, C)        │
    │                            │
    ↓                            │
[GeLU激活]                       │
    │                            │
    ↓                            │
[Dropout]                        │
    │                            │
    ↓                            │
[上投影]                         │
Linear(bottleneck_dim → in_dim)  │
    │                            │
    └────────────────────────────┘
              │
              ↓
        输出 (B, N, C)
```

## 安装要求

### 依赖项

```bash
torch >= 1.8.0
torchvision >= 0.9.0
```

### 安装PyTorch

```bash
# CPU版本
pip install torch torchvision

# GPU版本 (CUDA 11.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu111
```

## 快速开始

### 1. 导入模块

```python
from mona import Mona
```

### 2. 创建Mona适配器

```python
# 创建一个Mona适配器
# in_dim: 输入维度（通常与Transformer的隐藏维度相同）
# bottleneck_dim: 瓶颈维度（默认64，控制参数效率）
mona_layer = Mona(in_dim=384, bottleneck_dim=64)
```

### 3. 使用适配器

```python
import torch

# 准备输入数据
# x: (B, N, C) 其中 B=batch_size, N=H*W (序列长度), C=通道数
batch_size = 2
h, w = 14, 14  # 空间维度
n = h * w      # 序列长度
in_dim = 384   # 通道数
x = torch.randn(batch_size, n, in_dim)

# 前向传播
output = mona_layer(x, hw_shapes=(h, w))

# 输出形状与输入形状相同
assert output.shape == x.shape  # (2, 196, 384)
```

## API文档

### `Mona` 类

#### `__init__(self, in_dim, bottleneck_dim=64, dropout=0.1, init_scale=1e-6)`

创建Mona适配器实例。

**参数:**
- `in_dim` (int): 输入维度（通道数）
- `bottleneck_dim` (int, optional): 瓶颈维度，默认64
- `dropout` (float, optional): Dropout比率，默认0.1
- `init_scale` (float, optional): gamma参数的初始缩放值，默认1e-6

#### `forward(self, x, hw_shapes=None)`

前向传播。

**参数:**
- `x` (torch.Tensor): 输入张量，形状为 `(B, N, C)`
  - `B`: 批次大小
  - `N`: 序列长度（H * W）
  - `C`: 通道数（in_dim）
- `hw_shapes` (tuple, optional): 空间维度 `(H, W)`
  - 如果为None，将尝试从输入形状推断（假设为方形特征图）

**返回:**
- `output` (torch.Tensor): 输出张量，形状为 `(B, N, C)`

### `MonaOp` 类

#### `__init__(self, in_features)`

创建多认知卷积滤波器组。

**参数:**
- `in_features` (int): 输入通道数

#### `forward(self, x)`

前向传播。

**参数:**
- `x` (torch.Tensor): 输入张量，形状为 `(B, C, H, W)`

**返回:**
- `output` (torch.Tensor): 输出张量，形状为 `(B, C, H, W)`

## 使用示例

### 示例1: 基本使用

```python
from mona import Mona
import torch

# 创建适配器
mona = Mona(in_dim=384, bottleneck_dim=64)

# 准备输入
x = torch.randn(2, 196, 384)  # (batch, seq_len, dim)

# 前向传播
output = mona(x, hw_shapes=(14, 14))
print(f"输出形状: {output.shape}")  # (2, 196, 384)
```

### 示例2: 集成到Swin Transformer

```python
from mona import Mona
import torch
import torch.nn as nn

class SwinBlockWithMona(nn.Module):
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # ... 其他Swin Transformer组件 ...
        
        # 插入Mona适配器
        self.mona_after_attn = Mona(dim, bottleneck_dim=64)
        self.mona_after_mlp = Mona(dim, bottleneck_dim=64)
    
    def forward(self, x, H, W):
        # Self-Attention
        shortcut = x
        x = self.norm1(x)
        # ... 注意力计算 ...
        x = shortcut + x
        
        # Mona适配器1 (在注意力后)
        x = self.mona_after_attn(x, hw_shapes=(H, W))
        
        # MLP
        identity = x
        x = self.norm2(x)
        # ... MLP计算 ...
        x = identity + x
        
        # Mona适配器2 (在MLP后)
        x = self.mona_after_mlp(x, hw_shapes=(H, W))
        
        return x
```

### 示例3: 参数高效微调

```python
from mona import Mona
import torch
import torch.nn as nn

# 加载预训练模型
model = load_pretrained_model()

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 添加Mona适配器
model.adapter = Mona(in_dim=384, bottleneck_dim=64)

# 只训练适配器参数
optimizer = torch.optim.Adam(model.adapter.parameters(), lr=1e-3)

# 训练循环
for epoch in range(num_epochs):
    for batch in dataloader:
        # 前向传播
        x = batch['input']
        output = model(x)
        output = model.adapter(output, hw_shapes=(14, 14))
        
        # 计算损失
        loss = criterion(output, batch['target'])
        
        # 反向传播（只更新适配器参数）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 示例4: 统计参数量

```python
from mona import Mona

# 创建适配器
mona = Mona(in_dim=384, bottleneck_dim=64)

# 统计参数量
total_params = sum(p.numel() for p in mona.parameters())
trainable_params = sum(p.numel() for p in mona.parameters() if p.requires_grad)

print(f"总参数量: {total_params:,}")  # 约 50,000 参数
print(f"可训练参数量: {trainable_params:,}")  # 全部可训练

# 计算参数效率
full_linear_params = 384 * 384 * 4  # 假设两个全连接层
efficiency = total_params / full_linear_params
print(f"参数效率: {efficiency:.4%}")  # 约 8.5%
```

## 参数说明

### 关键参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_dim` | int | - | 输入维度（必需），通常与Transformer隐藏维度相同 |
| `bottleneck_dim` | int | 64 | 瓶颈维度，控制参数效率。越小参数量越少，但可能影响性能 |
| `dropout` | float | 0.1 | Dropout比率，防止过拟合 |
| `init_scale` | float | 1e-6 | gamma初始值，控制适配器初始影响程度 |

### 参数选择建议

- **in_dim**: 根据使用的Transformer模型选择
  - Swin-Tiny: 96
  - Swin-Small: 96
  - Swin-Base: 128
  - Swin-Large: 192

- **bottleneck_dim**: 
  - 较小模型: 32-64
  - 中等模型: 64-128
  - 大型模型: 128-256

- **dropout**: 
  - 小数据集: 0.1-0.2
  - 大数据集: 0.05-0.1

## 集成指南

### 在Swin Transformer中集成

1. **导入模块**
```python
from mona import Mona
```

2. **在SwinBlock中添加Mona**
```python
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ...):
        super().__init__()
        # ... 原有组件 ...
        
        # 添加Mona适配器
        self.mona_after_attn = Mona(dim, bottleneck_dim=64)
        self.mona_after_mlp = Mona(dim, bottleneck_dim=64)
```

3. **在前向传播中使用**
```python
def forward(self, x, H, W):
    # ... 注意力计算 ...
    x = self.mona_after_attn(x, hw_shapes=(H, W))
    
    # ... MLP计算 ...
    x = self.mona_after_mlp(x, hw_shapes=(H, W))
    
    return x
```

4. **冻结预训练参数**
```python
# 冻结所有非Mona参数
for name, param in model.named_parameters():
    if 'mona' not in name:
        param.requires_grad = False
```

### 在其他Transformer中集成

Mona适配器可以集成到任何Transformer架构中：

1. **ViT (Vision Transformer)**
```python
class ViTBlockWithMona(nn.Module):
    def __init__(self, dim, ...):
        # ... 原有组件 ...
        self.mona = Mona(dim, bottleneck_dim=64)
    
    def forward(self, x):
        # ... 注意力 ...
        x = self.mona(x, hw_shapes=(patch_h, patch_w))
        # ... MLP ...
        return x
```

2. **DeiT (Data-efficient Image Transformer)**
```python
# 类似ViT的集成方式
```

## 特性说明

### 1. 参数高效

- **参数量**: 相比全连接层，参数量减少约90%以上
- **计算量**: 计算开销较小，适合资源受限场景
- **内存占用**: 内存占用低，可以训练更大的批次

### 2. 多尺度特征提取

- **3x3卷积**: 捕获局部特征
- **5x5卷积**: 捕获中等范围特征
- **7x7卷积**: 捕获大范围特征
- **平均聚合**: 融合多尺度信息

### 3. 稳定训练

- **缩放LayerNorm**: 通过可学习的缩放因子控制适配器影响
- **跳跃连接**: 确保信息流畅通，防止梯度消失
- **渐进式激活**: 初始时适配器影响较小，训练过程中逐渐增大

### 4. 即插即用

- **无需修改原模型**: 可以直接插入到现有模型中
- **灵活集成**: 可以在任意位置插入适配器
- **兼容性好**: 与大多数Transformer架构兼容

## 常见问题

### Q1: 如何选择bottleneck_dim？

**A**: bottleneck_dim控制参数效率。建议：
- 小模型: 32-64
- 中等模型: 64-128
- 大模型: 128-256

可以通过实验选择最佳值。

### Q2: 为什么需要hw_shapes参数？

**A**: Mona适配器内部使用2D卷积，需要知道特征图的空间维度。如果输入是方形特征图，可以省略该参数，模块会自动推断。

### Q3: 如何冻结预训练模型参数？

**A**: 
```python
# 方法1: 冻结所有非Mona参数
for name, param in model.named_parameters():
    if 'mona' not in name:
        param.requires_grad = False

# 方法2: 只训练Mona参数
optimizer = torch.optim.Adam(
    [p for n, p in model.named_parameters() if 'mona' in n],
    lr=1e-3
)
```

### Q4: 适配器应该插入在哪里？

**A**: 通常插入在：
- 自注意力层之后
- MLP层之后

可以根据任务需求选择插入位置。

### Q5: 如何调整init_scale？

**A**: init_scale控制适配器的初始影响程度：
- 较小值 (1e-6): 适配器初始影响很小，训练更稳定
- 较大值 (1e-4): 适配器初始影响较大，可能收敛更快

建议使用默认值1e-6。

### Q6: 参数量是多少？

**A**: 参数量计算公式：
```
参数量 ≈ in_dim * bottleneck_dim * 2 + bottleneck_dim * bottleneck_dim * 4 + in_dim * 2
```

对于in_dim=384, bottleneck_dim=64:
- 约 50,000 参数
- 相比全连接层减少约91.5%

## 参考文献

- Mona: Multi-cognitive Visual Adapter for Parameter-Efficient Fine-tuning
- Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 许可证

请参考项目主目录的LICENSE文件。

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**最后更新**: 2024年

