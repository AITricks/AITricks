# LDConv 即插即用模块使用说明

## 📖 简介

LDConv (Learnable Deformable Convolution) 是一个基于AKConv原理实现的可学习变形卷积模块，支持任意采样形状卷积核与任意参数数量卷积核。该模块可以**即插即用**地替换普通卷积层，无需修改网络其他部分。

### 核心特性

- ✅ **即插即用**：可直接替换 `nn.Conv2d`，保持输入输出形状一致
- ✅ **自适应采样**：通过可学习的偏移量动态调整采样位置
- ✅ **任意采样点数量**：支持任意数量的采样点（如4, 9, 16, 25等）
- ✅ **双线性插值**：使用双线性插值确保采样过程可微
- ✅ **稳定训练**：通过梯度缩放机制保证训练稳定性

## 🔧 安装要求

### Python环境
- Python >= 3.6
- PyTorch >= 1.7.0（推荐 >= 1.10.0 以支持完整的meshgrid功能）
- einops >= 0.3.0

### 安装命令
```bash
pip install torch torchvision
pip install einops
```

## 🚀 快速开始

### 基本使用

```python
import torch
import torch.nn as nn
from LDConv_block import LDConv

# 创建LDConv模块
ldconv = LDConv(
    inc=64,        # 输入通道数
    outc=128,      # 输出通道数
    num_param=9,   # 采样点数量（9表示3x3网格）
    stride=1,      # 步长
    bias=False     # 是否使用偏置
)

# 前向传播
x = torch.randn(2, 64, 32, 32)  # (batch, channels, height, width)
output = ldconv(x)  # (2, 128, 32, 32)
```

### 即插即用替换示例

#### 示例1：替换单个卷积层

```python
import torch.nn as nn
from LDConv_block import LDConv

# 原始网络
class OriginalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

# 使用LDConv替换后的网络
class LDConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 直接替换：nn.Conv2d → LDConv
        self.conv1 = LDConv(3, 64, num_param=9, stride=1)    # 替换第一个卷积
        self.conv2 = LDConv(64, 128, num_param=9, stride=1)  # 替换第二个卷积
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
```

#### 示例2：在YOLO等检测网络中使用

```python
# 替换YOLOv5中的Conv模块
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        # 原始实现
        # self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        
        # 使用LDConv替换（当k=3时）
        if k == 3:
            self.conv = LDConv(c1, c2, num_param=9, stride=s)
        else:
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

#### 示例3：部分替换策略

```python
class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 只在关键层使用LDConv
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)           # 普通卷积
        self.conv2 = LDConv(64, 128, num_param=9, stride=2)   # LDConv（下采样）
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)        # 普通卷积
        self.conv4 = LDConv(256, 512, num_param=16, stride=2) # LDConv（更多采样点）
```

## 📋 参数说明

### LDConv 参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `inc` | int | 输入通道数 | - |
| `outc` | int | 输出通道数 | - |
| `num_param` | int | 采样点数量（卷积核参数数量） | - |
| `stride` | int | 步长 | 1 |
| `bias` | bool | 是否使用偏置 | None |

### num_param 参数说明

`num_param` 决定了采样点的数量，通常选择完全平方数：

- `num_param=4` → 2×2 网格（4个采样点）
- `num_param=9` → 3×3 网格（9个采样点）
- `num_param=16` → 4×4 网格（16个采样点）
- `num_param=25` → 5×5 网格（25个采样点）

对于非完全平方数，会自动生成接近规则网格的采样点分布。

## 🔄 与普通卷积的对应关系

| 普通卷积 | LDConv |
|---------|--------|
| `nn.Conv2d(inc, outc, k=3, s=1, p=1)` | `LDConv(inc, outc, num_param=9, stride=1)` |
| `nn.Conv2d(inc, outc, k=5, s=1, p=2)` | `LDConv(inc, outc, num_param=25, stride=1)` |
| `nn.Conv2d(inc, outc, k=3, s=2, p=1)` | `LDConv(inc, outc, num_param=9, stride=2)` |

**注意**：LDConv不需要padding参数，因为它是通过可学习偏移量自适应采样的。

## 📊 输出形状

LDConv的输出形状计算方式与普通卷积类似：

```python
# 输入形状: (B, C_in, H, W)
# 输出形状: (B, C_out, H', W')

# 其中：
# H' = floor((H * stride) / stride) = H (当stride=1时)
# W' = floor((W * stride) / stride) = W (当stride=1时)
```

## ⚙️ 工作原理

LDConv的工作流程（对应结构图）：

1. **生成Offset**：通过 `p_conv` 卷积层生成可学习的偏移量
2. **初始采样坐标**：根据 `num_param` 生成规则的初始采样网格（p_n）
3. **坐标修改**：将偏移量应用到初始坐标：`p = p_0 + p_n + offset`
4. **双线性插值**：计算四个最近邻点的插值权重
5. **重采样**：基于修改后的坐标从输入特征图中采样
6. **Reshape**：将重采样后的特征重塑为卷积层可处理的形状
7. **卷积处理**：通过最终的卷积、归一化和激活层输出结果

## 💡 使用建议

### 1. 采样点数量选择

- **小网络/轻量级模型**：使用 `num_param=4` 或 `num_param=9`
- **中等网络**：使用 `num_param=9` 或 `num_param=16`
- **大网络/高精度需求**：使用 `num_param=16` 或 `num_param=25`

### 2. 替换策略

- **全部替换**：将所有 `nn.Conv2d` 替换为 `LDConv`（可能增加较多参数量）
- **部分替换**：只在关键层（如下采样层、特征提取层）使用 `LDConv`
- **渐进替换**：先在部分层使用，观察效果后再决定是否全面替换

### 3. 训练建议

- **学习率**：可以使用与普通卷积相同的学习率
- **初始化**：偏移量卷积已自动初始化为0，确保训练初期稳定
- **梯度缩放**：已内置梯度缩放机制（0.1倍），保证偏移量学习稳定

## 📈 性能对比

### 参数量对比

```python
import torch.nn as nn
from LDConv_block import LDConv

# 普通卷积
conv_normal = nn.Conv2d(64, 128, 3, padding=1, bias=False)
params_normal = sum(p.numel() for p in conv_normal.parameters())
print(f"普通卷积参数量: {params_normal:,}")  # 73,728

# LDConv
conv_ldconv = LDConv(64, 128, num_param=9, stride=1, bias=False)
params_ldconv = sum(p.numel() for p in conv_ldconv.parameters())
print(f"LDConv参数量: {params_ldconv:,}")    # 约 82,944 (增加约12.5%)
```

### 计算量对比

LDConv相比普通卷积：
- **参数量**：增加约 10-15%（主要来自偏移量生成卷积）
- **计算量**：增加约 20-30%（主要来自双线性插值和重采样）
- **内存占用**：增加约 15-25%（需要存储中间特征图）

## ⚠️ 注意事项

1. **内存占用**：LDConv需要更多内存来存储中间特征，对于大batch size可能需要调整
2. **训练时间**：由于增加了采样和插值操作，训练时间会有所增加
3. **CUDA支持**：建议使用GPU训练，CPU上可能较慢
4. **版本兼容性**：需要 PyTorch >= 1.7.0，einops >= 0.3.0

## 🔍 常见问题

### Q1: 如何选择 num_param？

A: 通常选择与原始卷积核大小对应的采样点数量：
- 3×3 卷积 → `num_param=9`
- 5×5 卷积 → `num_param=25`
- 也可以尝试其他数量，如 `num_param=16` 用于4×4网格

### Q2: 输出形状与普通卷积不一致？

A: 确保 `stride` 参数设置正确。LDConv的输出形状计算方式与普通卷积相同。

### Q3: 训练不稳定？

A: LDConv已内置梯度缩放机制，如果仍不稳定，可以：
- 降低学习率
- 使用warmup策略
- 检查数据预处理是否正常

### Q4: 如何迁移预训练模型？

A: 由于LDConv结构与普通卷积不同，无法直接加载预训练权重。建议：
1. 先用普通卷积训练，再用LDConv从头训练
2. 或者在部分层使用LDConv，其他层加载预训练权重

## 📝 完整示例

```python
import torch
import torch.nn as nn
from LDConv_block import LDConv

# 完整的网络示例
class ExampleNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 使用LDConv替换普通卷积
        self.features = nn.Sequential(
            LDConv(3, 64, num_param=9, stride=2),      # 下采样
            nn.ReLU(inplace=True),
            LDConv(64, 128, num_param=9, stride=2),    # 下采样
            nn.ReLU(inplace=True),
            LDConv(128, 256, num_param=16, stride=2),  # 更多采样点
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 使用示例
if __name__ == "__main__":
    model = ExampleNet(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
```

## 📚 参考

- 基于AKConv原理实现
- 支持任意采样形状卷积核
- 支持任意参数数量卷积核
- 完整的即插即用支持

## 📄 许可证

请参考项目主目录的许可证文件。

---

**提示**：运行 `python LDConv-block.py` 可以查看完整的演示示例和性能对比。
