# Strip R-CNN 即插即用模块说明

## 📋 概述

本文档说明了从 Strip R-CNN 论文中提取的即插即用模块,这些模块已保存在 `strip_modules_plugandplay.py` 文件中。

## 🎯 提取的模块

根据论文结构图分析,我们提取了以下4个即插即用模块:

### 1. **StripBlock** - 条形卷积块 ⭐核心模块

**来源**: Figure 4 - Strip Module

**结构**:
```
Input → Square Conv (5×5) → H_Strip Conv (1×19) → V_Strip Conv (19×1) → PW Conv (1×1) → Attention Weights
                                                                                              ↓
Output ← Element-wise Multiply ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

**参数**:
- `dim`: 输入输出通道数
- `strip_kernel_size`: 条形卷积核大小 (默认19)

**用途**: 
- 捕获长条形目标特征 (如遥感图像中的道路、船舶、飞机等)
- 可直接插入任何CNN网络的特征提取层

**使用示例**:
```python
from strip_modules_plugandplay import StripBlock

# 在ResNet的某一层后添加
strip_layer = StripBlock(dim=256, strip_kernel_size=19)
enhanced_features = strip_layer(features)  # features: [B, 256, H, W]
```

---

### 2. **CenterPooling** - 中心池化空间注意力

**来源**: Figure 3(c) - Spatial Attention in Strip R-CNN

**结构**:
```
                    ┌→ Conv → MaxPool(dim=W) → Expand ┐
Input → Split →     │                                  ├→ Add → Conv → Output
                    └→ Conv → MaxPool(dim=H) → Expand ┘
                                                        ↓
                                                   + Residual
```

**参数**:
- `in_channels`: 输入通道数
- `mid_channels`: 中间层通道数
- `out_channels`: 输出通道数

**用途**:
- 空间注意力机制
- 在水平和垂直方向上捕获全局信息

**使用示例**:
```python
from strip_modules_plugandplay import CenterPooling

# 在FPN特征融合前使用
spatial_attn = CenterPooling(in_channels=256, mid_channels=128, out_channels=256)
attended_features = spatial_attn(fpn_features)
```

---

### 3. **StripAttention** - 完整条形注意力模块

**来源**: Figure 4 - 完整的Strip Block结构

**结构**:
```
Input → Proj (1×1) → GELU → StripBlock → Proj (1×1) → Add with Residual → Output
  ↓                                                           ↑
  └←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←┘
```

**参数**:
- `dim`: 特征维度
- `strip_kernel_size`: 条形卷积核大小

**用途**:
- StripBlock的增强版本
- 包含投影层和残差连接
- 更强的特征表达能力

---

### 4. **StripEnhancedBlock** - 增强条形块 (含FFN)

**来源**: Figure 4 - 完整的Strip Block + FFN结构

**结构**:
```
Input → Norm → StripAttention → LayerScale → Add ┐
  ↓                                               ↓
  └→ Norm → FFN (MLP) → LayerScale → Add ←←←←←←←←┘
                                      ↓
                                   Output
```

**参数**:
- `dim`: 特征维度
- `mlp_ratio`: FFN扩展比例 (默认4.0)
- `strip_kernel_size`: 条形卷积核大小
- `drop`: Dropout比例

**用途**:
- 完整的Transformer-like块
- 性能最强的即插即用模块
- 适合作为backbone的基础块

---

## 📊 模块对比

| 模块 | 参数量 (C=256) | 计算复杂度 | 适用场景 | 推荐度 |
|------|---------------|-----------|---------|--------|
| StripBlock | ~66K | 低 | 轻量级特征增强 | ⭐⭐⭐⭐ |
| CenterPooling | ~394K | 中 | 空间注意力 | ⭐⭐⭐ |
| StripAttention | ~197K | 中 | 通用特征增强 | ⭐⭐⭐⭐⭐ |
| StripEnhancedBlock | ~1.3M | 高 | Backbone基础块 | ⭐⭐⭐⭐⭐ |

## 🚀 快速开始

### 安装依赖
```bash
pip install torch torchvision
```

### 运行测试
```bash
python strip_modules_plugandplay.py
```

### 集成到现有网络

#### 示例1: 在ResNet中使用
```python
import torch.nn as nn
from strip_modules_plugandplay import StripBlock

class ResNetWithStrip(nn.Module):
    def __init__(self, resnet_backbone):
        super().__init__()
        self.backbone = resnet_backbone
        # 在layer3后添加StripBlock
        self.strip_enhance = StripBlock(dim=1024, strip_kernel_size=19)
    
    def forward(self, x):
        # ResNet前向传播
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        
        # 应用Strip增强
        x = self.strip_enhance(x)
        
        x = self.backbone.layer4(x)
        return x
```

#### 示例2: 在FPN中使用
```python
from strip_modules_plugandplay import CenterPooling, StripAttention

class EnhancedFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # 为每个FPN层添加增强模块
        self.strip_layers = nn.ModuleList([
            StripAttention(dim=out_channels) 
            for _ in in_channels_list
        ])
        self.spatial_attn = CenterPooling(out_channels, out_channels//2, out_channels)
    
    def forward(self, features):
        # features: list of [P2, P3, P4, P5]
        enhanced = []
        for feat, strip_layer in zip(features, self.strip_layers):
            feat = strip_layer(feat)
            feat = self.spatial_attn(feat)
            enhanced.append(feat)
        return enhanced
```

#### 示例3: 构建自定义Backbone
```python
from strip_modules_plugandplay import StripEnhancedBlock

class StripBackbone(nn.Module):
    def __init__(self, in_channels=3, embed_dims=[64, 128, 256, 512], depths=[2, 2, 6, 2]):
        super().__init__()
        self.stages = nn.ModuleList()
        
        for i, (dim, depth) in enumerate(zip(embed_dims, depths)):
            # Patch Embedding
            if i == 0:
                patch_embed = nn.Conv2d(in_channels, dim, kernel_size=7, stride=4, padding=3)
            else:
                patch_embed = nn.Conv2d(embed_dims[i-1], dim, kernel_size=3, stride=2, padding=1)
            
            # Strip Enhanced Blocks
            blocks = nn.Sequential(*[
                StripEnhancedBlock(dim, mlp_ratio=4.0, strip_kernel_size=19)
                for _ in range(depth)
            ])
            
            self.stages.append(nn.Sequential(patch_embed, blocks))
    
    def forward(self, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
```

## 📈 性能特点

### 优势
1. **针对长条形目标优化**: 使用1×19和19×1的条形卷积,特别适合遥感图像
2. **即插即用**: 无需修改网络架构,可直接插入现有模型
3. **轻量级**: StripBlock参数量小,计算开销低
4. **灵活性**: 可调节strip_kernel_size适应不同尺度的目标

### 适用场景
- ✅ 遥感图像目标检测 (道路、船舶、飞机)
- ✅ 文本检测 (长条形文本行)
- ✅ 医学图像分析 (血管、神经纤维)
- ✅ 工业缺陷检测 (裂纹、划痕)

## 🔬 论文引用

如果使用这些模块,请引用原论文:

```bibtex
@article{stripcnn,
  title={Strip R-CNN: Rethinking the Spatial Encoding for Oriented Object Detection},
  author={...},
  journal={...},
  year={2024}
}
```

## 📝 注意事项

1. **strip_kernel_size选择**: 
   - 小目标: 11-15
   - 中等目标: 17-21 (默认19)
   - 大目标: 23-27

2. **内存占用**: StripEnhancedBlock包含FFN,内存占用较大,注意batch size调整

3. **训练技巧**: 
   - 建议使用Layer Scale (已内置)
   - 可配合DropPath使用
   - 学习率可设置为backbone的0.1倍

## 🛠️ 自定义修改

### 修改条形卷积核大小
```python
# 针对更大的目标
strip_block = StripBlock(dim=256, strip_kernel_size=31)

# 针对更小的目标  
strip_block = StripBlock(dim=256, strip_kernel_size=11)
```

### 修改FFN扩展比例
```python
# 更强的表达能力
enhanced_block = StripEnhancedBlock(dim=256, mlp_ratio=6.0)

# 更轻量级
enhanced_block = StripEnhancedBlock(dim=256, mlp_ratio=2.0)
```

---

**创建日期**: 2025-12-09  
**文件位置**: `strip_modules_plugandplay.py`  
**测试状态**: ✅ 已通过单元测试
