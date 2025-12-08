# 几何自注意力模块 (Geo-GSA Plug)

即插即用的几何先验 + 几何自注意力模块，从 DFormerv2 中抽取的核心组件。

## 📋 模块简介

本模块实现了融合深度信息的几何自注意力机制，适用于 RGB-D 图像处理任务（如深度估计、语义分割等）。模块将深度图的几何信息作为先验知识融入自注意力计算中，提升模型对空间几何关系的理解能力。

## 🔧 主要组件

### 1. `GeoPriorGen` - 几何先验生成器
根据深度图和空间位置生成几何先验信息，包括：
- **位置先验**：基于像素空间距离的衰减掩码
- **深度先验**：基于深度差异的衰减掩码
- **旋转位置编码（RoPE）**：用于编码位置信息

### 2. `Full_GSA` - 完整 2D 几何自注意力
在完整的 2D 空间上进行自注意力计算，计算复杂度为 O(H²W²)。

### 3. `Decomposed_GSA` - 分解式几何自注意力
将 2D 注意力分解为沿宽度（W）和高度（H）两个方向的一维注意力，计算复杂度降低到 O(H²W + HW²)。

### 4. `RGBD_Block` - RGB-D 处理块
完整的即插即用模块，包含：
- 几何先验生成
- 几何自注意力
- 前馈网络（FFN）
- 残差连接和 LayerNorm

## 📥 输入/输出格式

- **输入特征 x**: `(B, H, W, C)` - 主干网络提取的 RGB 特征
- **深度图 x_e**: `(B, 1, H, W)` - 对应的深度图
- **输出特征**: `(B, H, W, C)` - 增强后的特征

## 🚀 快速开始

### 基本使用

```python
import torch
from geo_gsa_plug import RGBD_Block

# 准备输入
B, H, W, C = 2, 64, 64, 256  # 批次大小、高度、宽度、通道数
x = torch.randn(B, H, W, C)  # RGB 特征 (B, H, W, C)
x_e = torch.randn(B, 1, H, W)  # 深度图 (B, 1, H, W)

# 创建模块
block = RGBD_Block(
    split_or_not=False,  # False: 使用完整 2D 注意力, True: 使用分解式注意力
    embed_dim=256,  # 嵌入维度
    num_heads=8,  # 注意力头数
    ffn_dim=1024,  # FFN 扩展维度（通常为 embed_dim 的 4 倍）
    drop_path=0.1,  # 随机深度比例
    layerscale=False,  # 是否启用 LayerScale
)

# 前向传播
output = block(x, x_e, split_or_not=False)
print(f"输出形状: {output.shape}")  # (B, H, W, C)
```

### 完整示例

```python
import torch
from geo_gsa_plug import RGBD_Block

# 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"

# 参数设置
embed_dim = 256
num_heads = 8
ffn_dim = 4 * embed_dim
H, W = 64, 64

# 创建模块
block = RGBD_Block(
    split_or_not=False,
    embed_dim=embed_dim,
    num_heads=num_heads,
    ffn_dim=ffn_dim,
    drop_path=0.1,
    layerscale=True,
    layer_init_values=1e-5,
    init_value=2.0,
    heads_range=4.0,
).to(device)

# 准备数据
x = torch.randn(2, H, W, embed_dim, device=device)
depth = torch.randn(2, 1, H, W, device=device)

# 前向传播
with torch.no_grad():
    output = block(x, depth)
    print(f"输入: {x.shape}, 输出: {output.shape}")
```

## ⚙️ 参数说明

### `RGBD_Block` 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `split_or_not` | bool | - | 是否使用分解式注意力（True: 分解式, False: 完整 2D） |
| `embed_dim` | int | - | 嵌入维度（通道数） |
| `num_heads` | int | - | 注意力头数 |
| `ffn_dim` | int | - | 前馈网络扩展维度 |
| `drop_path` | float | 0.0 | 随机深度比例 |
| `layerscale` | bool | False | 是否启用 LayerScale |
| `layer_init_values` | float | 1e-5 | LayerScale 初始化因子 |
| `init_value` | float | 2.0 | 几何衰减初始值 |
| `heads_range` | float | 4.0 | 不同注意力头之间的衰减范围 |

## 💡 使用建议

1. **选择注意力模式**：
   - `split_or_not=False`: 适用于较小的特征图（如 64×64），计算完整 2D 注意力
   - `split_or_not=True`: 适用于较大的特征图，使用分解式注意力降低计算量

2. **嵌入维度设置**：
   - 通常与主干网络的通道数一致
   - 确保能被 `num_heads` 整除

3. **FFN 维度**：
   - 通常设置为 `embed_dim` 的 4 倍

4. **训练技巧**：
   - 可以启用 `layerscale` 提升训练稳定性
   - 使用 `drop_path` 进行正则化

## 📚 相关技术

- **旋转位置编码（RoPE）**: 将位置信息编码到特征向量中
- **局部位置编码（LePE）**: 通过深度可分离卷积增强局部位置信息
- **几何先验**: 融合空间位置和深度信息的先验知识

## 📝 注意事项

- 输入特征需要是 `(B, H, W, C)` 格式，而不是常见的 `(B, C, H, W)` 格式
- 深度图会自动插值到与特征图相同的尺寸
- 模块支持批处理，可以同时处理多个样本

## 🔗 参考

本模块基于 DFormerv2 论文实现，更多细节请参考原论文。

