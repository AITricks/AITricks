# ASSA模块使用说明

## 简介
ASSA (Adaptive Sparse Self-Attention) 是自适应稀疏自注意力模块，可用于图像恢复等任务。

## 快速开始

### 安装依赖
```bash
pip install torch einops
```

### 基本使用

```python
from assa_module import ASSA, window_partition, window_reverse
import torch

# 1. 创建ASSA模块
assa = ASSA(
    dim=64,              # 特征维度
    win_size=(8, 8),     # 窗口大小
    num_heads=8,         # 注意力头数
    attn_drop=0.1,       # 注意力dropout
    proj_drop=0.1        # 投影dropout
)

# 2. 准备输入数据
B, H, W, C = 2, 32, 32, 64  # batch, height, width, channels
x = torch.randn(B, H, W, C)

# 3. 窗口分割
x_windows = window_partition(x, win_size=8)  # [B*num_windows, win_size, win_size, C]
x_windows = x_windows.view(-1, 8*8, C)       # [B*num_windows, win_size^2, C]

# 4. 前向传播
output = assa(x_windows)  # [B*num_windows, win_size^2, C]

# 5. 窗口还原（可选）
output = output.view(-1, 8, 8, C)
output = window_reverse(output, win_size=8, H=H, W=W)  # [B, H, W, C]
```

## 参数说明

### ASSA参数
- `dim`: 输入特征维度
- `win_size`: 窗口大小，元组格式 (H, W)
- `num_heads`: 多头注意力的头数
- `token_projection`: 投影方式，默认 'linear'
- `qkv_bias`: 是否使用QKV偏置，默认 True
- `qk_scale`: QK缩放因子，默认 None（自动计算）
- `attn_drop`: 注意力dropout率，默认 0.0
- `proj_drop`: 投影dropout率，默认 0.0

## 特性

- **自适应融合**: 自动融合动态稀疏注意力(DSA)和静态稀疏注意力(SSA)
- **相对位置编码**: 支持窗口内的相对位置偏置
- **即插即用**: 可轻松集成到现有模型中

## 在你的网络中集成

```python
from assa_module import ASSA, window_partition, window_reverse
import torch

# 创建ASSA模块
attn = ASSA(dim=64, win_size=(8, 8), num_heads=8, 
            attn_drop=0.1, proj_drop=0.1)

# 准备输入数据（假设已有特征图）
B, H, W, C = 2, 32, 32, 64
x = torch.randn(B, H, W, C)

# 窗口分割
x_windows = window_partition(x, win_size=8)
x_windows = x_windows.view(-1, 8*8, C)

# 前向传播
out = attn(x_windows, mask=None)

# 窗口还原
out = out.view(-1, 8, 8, C)
out = window_reverse(out, win_size=8, H=H, W=W)
```

**小贴士**：若你在现有模型的 `WindowAttention` 位置替换为本模块，也能复用相同的窗口分割逻辑和相对位置编码。ASSA模块会自动处理相对位置偏置和注意力掩码。

## 测试

运行测试函数验证模块功能：
```bash
python assa_module.py
```

## 注意事项

- 输入格式：`[B*num_windows, N, C]`，其中N是窗口内的token数
- 默认使用 `dilation_rate=1`，如需使用其他值请先测试
- 模块会自动处理相对位置编码和注意力掩码

