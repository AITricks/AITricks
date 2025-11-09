# DAT 即插即用模块使用说明

本文档介绍如何使用 DAT（Dual Aggregation Transformer）的即插即用模块。

## 1. 安装命令

### 环境要求
- Python 3.8+
- PyTorch 1.8.0+
- CUDA（GPU加速，可选）

### 安装步骤

```bash
# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision

# 安装依赖
pip install einops

# 如果使用完整 DAT 项目
pip install -r requirements.txt
```

### 最小依赖
只需要以下两个包即可使用模块：
- `torch` - PyTorch 深度学习框架
- `einops` - 用于张量重排操作

```bash
pip install torch einops
```

---

## 2. 基本使用示例

### 2.1 使用 SGFN（空间门控前馈网络）

```python
import torch
from test_dat_plug_and_play import SGFN_Wrapper

# 创建模型
model = SGFN_Wrapper(
    channel=64,              # 输入通道数
    hidden_features=256,     # 隐藏层特征数（可选，默认 channel*4）
    drop=0.0                 # Dropout 比例（可选）
)

# 输入数据：格式为 [B, C, H, W]
input_tensor = torch.randn(1, 64, 256, 256)  # [batch, channels, height, width]

# 前向传播
model.eval()  # 评估模式
output = model(input_tensor)

# 输出形状与输入相同
print(f"输入形状: {input_tensor.shape}")  # torch.Size([1, 64, 256, 256])
print(f"输出形状: {output.shape}")        # torch.Size([1, 64, 256, 256])
```

### 2.2 使用自适应通道注意力

```python
import torch
from test_dat_plug_and_play import Adaptive_Channel_Attention_Wrapper

# 创建模型
model = Adaptive_Channel_Attention_Wrapper(
    channel=64,        # 输入通道数
    num_heads=8,       # 注意力头数
    qkv_bias=False,    # 是否使用 QKV 偏置（可选）
    attn_drop=0.0,     # 注意力 Dropout（可选）
    proj_drop=0.0      # 投影层 Dropout（可选）
)

# 输入数据
input_tensor = torch.randn(1, 64, 256, 256)

# 前向传播
model.eval()
output = model(input_tensor)

print(f"输出形状: {output.shape}")  # torch.Size([1, 64, 256, 256])
```

### 2.3 使用自适应空间注意力

```python
import torch
from test_dat_plug_and_play import Adaptive_Spatial_Attention_Wrapper

# 创建模型
model = Adaptive_Spatial_Attention_Wrapper(
    channel=64,           # 输入通道数
    num_heads=8,          # 注意力头数
    reso=256,             # 特征图分辨率
    split_size=[8, 8],    # 窗口大小 [H, W]
    shift_size=[1, 2],    # 窗口 shift 大小
    rg_idx=0,             # 残差组索引
    b_idx=0               # 块索引
)

# 输入数据
input_tensor = torch.randn(1, 64, 256, 256)

# 前向传播
model.eval()
output = model(input_tensor)

print(f"输出形状: {output.shape}")  # torch.Size([1, 64, 256, 256])
```

### 2.4 完整示例：构建简单网络

```python
import torch
import torch.nn as nn
from test_dat_plug_and_play import (
    SGFN_Wrapper,
    Adaptive_Channel_Attention_Wrapper,
    Adaptive_Spatial_Attention_Wrapper
)

class SimpleDATBlock(nn.Module):
    """简单的 DAT 模块组合"""
    def __init__(self, channel=64):
        super().__init__()
        # 通道注意力
        self.channel_attn = Adaptive_Channel_Attention_Wrapper(
            channel=channel, 
            num_heads=8
        )
        # 空间注意力
        self.spatial_attn = Adaptive_Spatial_Attention_Wrapper(
            channel=channel,
            num_heads=8,
            reso=64,
            split_size=[8, 8],
            shift_size=[1, 2]
        )
        # 前馈网络
        self.ffn = SGFN_Wrapper(
            channel=channel,
            hidden_features=channel * 4
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(channel)
        self.norm2 = nn.LayerNorm(channel)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 转换为序列格式
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        # 通道注意力 + 残差
        x_seq = x_seq + self.channel_attn(x)
        x_seq = self.norm1(x_seq)
        
        # 空间注意力 + 残差
        x_seq = x_seq + self.spatial_attn(x)
        x_seq = self.norm2(x_seq)
        
        # 前馈网络 + 残差
        x = x + self.ffn(x)
        
        return x

# 使用示例
model = SimpleDATBlock(channel=64)
input_tensor = torch.randn(1, 64, 64, 64)
model.eval()
output = model(input_tensor)
print(f"输出形状: {output.shape}")  # torch.Size([1, 64, 64, 64])
```

---

## 3. 核心参数说明

### 3.1 SGFN_Wrapper 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `channel` | int | 必需 | 输入/输出通道数 |
| `hidden_features` | int | `channel * 4` | 隐藏层特征数，控制模型容量 |
| `act_layer` | nn.Module | `nn.GELU` | 激活函数 |
| `drop` | float | `0.0` | Dropout 比例 |

### 3.2 Adaptive_Channel_Attention_Wrapper 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `channel` | int | 必需 | 输入/输出通道数 |
| `num_heads` | int | `8` | 注意力头数，必须能被 channel 整除 |
| `qkv_bias` | bool | `False` | 是否在 QKV 线性层使用偏置 |
| `qk_scale` | float | `None` | QK 缩放因子，None 时自动计算 |
| `attn_drop` | float | `0.0` | 注意力 Dropout 比例 |
| `proj_drop` | float | `0.0` | 输出投影层 Dropout 比例 |

### 3.3 Adaptive_Spatial_Attention_Wrapper 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `channel` | int | 必需 | 输入/输出通道数 |
| `num_heads` | int | `8` | 注意力头数 |
| `reso` | int | `64` | 特征图分辨率（用于预计算掩码） |
| `split_size` | list | `[8, 8]` | 窗口大小 `[H_sp, W_sp]` |
| `shift_size` | list | `[1, 2]` | 窗口 shift 大小 `[shift_h, shift_w]` |
| `rg_idx` | int | `0` | 残差组索引，控制是否使用 shift |
| `b_idx` | int | `0` | 块索引，控制是否使用 shift |
| `qkv_bias` | bool | `False` | QKV 偏置 |
| `attn_drop` | float | `0.0` | 注意力 Dropout |
| `proj_drop` | float | `0.0` | 投影层 Dropout |

**重要参数说明：**

- **`split_size`**: 窗口大小，控制局部注意力的范围。较小值（如 [4,4]）计算更快但感受野小，较大值（如 [16,16]）感受野大但计算更慢。
- **`shift_size`**: 必须小于 `split_size`，用于增加感受野。当 `rg_idx` 和 `b_idx` 满足特定条件时会启用。
- **`rg_idx` 和 `b_idx`**: 控制是否使用 shift 窗口。当 `(rg_idx % 2 == 0 and b_idx > 0 and (b_idx - 2) % 4 == 0)` 或 `(rg_idx % 2 != 0 and b_idx % 4 == 0)` 时启用 shift。

---

## 4. 关键要点

### 4.1 输入格式

所有包装类（Wrapper）都接受标准的图像张量格式：

```python
# 输入格式：[Batch, Channels, Height, Width]
input_tensor = torch.randn(B, C, H, W)

# 例如：
input_tensor = torch.randn(1, 64, 256, 256)  # 1个样本，64通道，256x256分辨率
```

**重要：**
- 输入和输出形状完全相同 `[B, C, H, W]`
- 支持任意大小的 H 和 W（不需要是窗口大小的倍数）
- 内部会自动处理 padding 和 reshape

### 4.2 窗口操作

#### 窗口分割
- `split_size=[8, 8]` 表示将图像分割成 8×8 的窗口
- 每个窗口内部计算自注意力
- 自动处理边界填充，确保图像能被窗口大小整除

#### Shift 窗口
- 用于增加感受野，让不同窗口之间能够交互
- 通过 `shift_size` 控制 shift 的大小
- 需要配合掩码使用，避免边界产生不合理注意力

#### 使用建议
```python
# 小图像（64x64）：使用小窗口
spatial_attn = Adaptive_Spatial_Attention_Wrapper(
    channel=64,
    split_size=[4, 4],    # 小窗口
    shift_size=[1, 1],
    reso=64
)

# 中等图像（256x256）：使用中等窗口
spatial_attn = Adaptive_Spatial_Attention_Wrapper(
    channel=64,
    split_size=[8, 8],    # 中等窗口
    shift_size=[2, 2],
    reso=256
)

# 大图像（512x512）：使用大窗口
spatial_attn = Adaptive_Spatial_Attention_Wrapper(
    channel=64,
    split_size=[16, 16],  # 大窗口
    shift_size=[4, 4],
    reso=512
)
```

### 4.3 替换方法

#### 在现有网络中替换注意力模块

**替换标准自注意力：**
```python
# 原来的代码
self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8)

# 替换为 DAT 通道注意力
from test_dat_plug_and_play import Adaptive_Channel_Attention_Wrapper
self.attention = Adaptive_Channel_Attention_Wrapper(channel=64, num_heads=8)
```

**替换空间注意力：**
```python
# 原来的代码
self.spatial_attn = SomeSpatialAttention(dim=64)

# 替换为 DAT 空间注意力
from test_dat_plug_and_play import Adaptive_Spatial_Attention_Wrapper
self.spatial_attn = Adaptive_Spatial_Attention_Wrapper(
    channel=64,
    num_heads=8,
    split_size=[8, 8],
    shift_size=[1, 2]
)
```

**替换前馈网络：**
```python
# 原来的代码
self.ffn = nn.Sequential(
    nn.Linear(64, 256),
    nn.GELU(),
    nn.Linear(256, 64)
)

# 替换为 DAT SGFN
from test_dat_plug_and_play import SGFN_Wrapper
self.ffn = SGFN_Wrapper(channel=64, hidden_features=256)
```

#### 完整替换示例

```python
import torch.nn as nn
from test_dat_plug_and_play import (
    Adaptive_Channel_Attention_Wrapper,
    Adaptive_Spatial_Attention_Wrapper,
    SGFN_Wrapper
)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用 DAT 模块
        self.channel_attn = Adaptive_Channel_Attention_Wrapper(
            channel=64, num_heads=8
        )
        self.spatial_attn = Adaptive_Spatial_Attention_Wrapper(
            channel=64,
            num_heads=8,
            split_size=[8, 8],
            shift_size=[1, 2]
        )
        self.ffn = SGFN_Wrapper(channel=64)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = x + self.channel_attn(x)  # 残差连接
        x = x + self.spatial_attn(x)  # 残差连接
        x = x + self.ffn(x)           # 残差连接
        return x
```

### 4.4 训练和推理模式

```python
# 训练模式
model.train()
output = model(input_tensor)

# 推理模式（推荐用于测试）
model.eval()
with torch.no_grad():  # 节省内存
    output = model(input_tensor)
```

**注意：** 当 batch_size=1 时，建议使用 `model.eval()` 避免 BatchNorm 出现问题。

---

## 5. 测试命令

### 5.1 运行内置测试

文件 `test_dat_plug_and_play.py` 包含了完整的测试代码：

```bash
# 运行所有测试
python test_dat_plug_and_play.py
```

测试会依次验证：
- SGFN 模块
- 自适应通道注意力模块
- 自适应空间注意力模块

### 5.2 单独测试模块

```python
# 测试 SGFN
python -c "
import torch
from test_dat_plug_and_play import SGFN_Wrapper
model = SGFN_Wrapper(channel=32, hidden_features=128)
input_tensor = torch.randn(1, 32, 256, 256)
model.eval()
output = model(input_tensor)
print(f'测试通过！输入: {input_tensor.shape}, 输出: {output.shape}')
"

# 测试通道注意力
python -c "
import torch
from test_dat_plug_and_play import Adaptive_Channel_Attention_Wrapper
model = Adaptive_Channel_Attention_Wrapper(channel=32, num_heads=8)
input_tensor = torch.randn(1, 32, 256, 256)
model.eval()
output = model(input_tensor)
print(f'测试通过！输入: {input_tensor.shape}, 输出: {output.shape}')
"

# 测试空间注意力
python -c "
import torch
from test_dat_plug_and_play import Adaptive_Spatial_Attention_Wrapper
model = Adaptive_Spatial_Attention_Wrapper(
    channel=32, num_heads=8, reso=256, 
    split_size=[8, 8], shift_size=[1, 2]
)
input_tensor = torch.randn(1, 32, 256, 256)
model.eval()
output = model(input_tensor)
print(f'测试通过！输入: {input_tensor.shape}, 输出: {output.shape}')
"
```

### 5.3 性能测试

```python
import torch
import time
from test_dat_plug_and_play import Adaptive_Spatial_Attention_Wrapper

# 创建模型
model = Adaptive_Spatial_Attention_Wrapper(
    channel=64,
    num_heads=8,
    reso=256,
    split_size=[8, 8],
    shift_size=[1, 2]
)
model.eval()

# 测试数据
input_tensor = torch.randn(1, 64, 256, 256)

# 预热
for _ in range(10):
    _ = model(input_tensor)

# 计时
torch.cuda.synchronize() if torch.cuda.is_available() else None
start_time = time.time()
for _ in range(100):
    output = model(input_tensor)
torch.cuda.synchronize() if torch.cuda.is_available() else None
end_time = time.time()

avg_time = (end_time - start_time) / 100
print(f"平均推理时间: {avg_time*1000:.2f} ms")
print(f"输出形状: {output.shape}")
```

### 5.4 验证输出形状

```python
import torch
from test_dat_plug_and_play import (
    SGFN_Wrapper,
    Adaptive_Channel_Attention_Wrapper,
    Adaptive_Spatial_Attention_Wrapper
)

def test_shape_preservation():
    """测试所有模块是否保持输入输出形状一致"""
    input_shapes = [
        (1, 32, 64, 64),
        (1, 64, 128, 128),
        (1, 128, 256, 256),
        (2, 64, 256, 256),  # batch_size > 1
    ]
    
    modules = [
        ("SGFN", SGFN_Wrapper(channel=64)),
        ("ChannelAttention", Adaptive_Channel_Attention_Wrapper(channel=64, num_heads=8)),
        ("SpatialAttention", Adaptive_Spatial_Attention_Wrapper(
            channel=64, num_heads=8, reso=256, split_size=[8, 8]
        )),
    ]
    
    for module_name, module in modules:
        module.eval()
        print(f"\n测试 {module_name}:")
        for shape in input_shapes:
            if shape[1] != 64:  # 跳过通道数不匹配的测试
                continue
            input_tensor = torch.randn(*shape)
            output = module(input_tensor)
            assert output.shape == input_tensor.shape, \
                f"{module_name} 形状不匹配: {input_tensor.shape} -> {output.shape}"
            print(f"  ✓ {shape} -> {output.shape}")

if __name__ == "__main__":
    test_shape_preservation()
    print("\n所有测试通过！")
```