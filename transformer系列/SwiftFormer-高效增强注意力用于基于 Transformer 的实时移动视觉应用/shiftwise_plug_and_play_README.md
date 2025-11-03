# Shift-Wise Convolution 即插即用模块使用说明

## 文件说明

`shiftwise_plug_and_play.py` - 独立的即插即用模块，可直接替换标准Conv2d层

## 环境要求

- Python 3.6+
- PyTorch 1.0+ (兼容torchv5环境)
- numpy

## 在Anaconda torchv5环境中运行

### 1. 激活环境

```bash
conda activate torchv5
```

### 2. 安装依赖（如果未安装）

```bash
pip install numpy
```

### 3. 运行测试

```bash
python shiftwise_plug_and_play.py
```

## 使用方法

### 基础用法

```python
from shiftwise_plug_and_play import ShifthWiseConv2dImplicit
import torch
import torch.nn as nn

# 创建模块
conv = ShifthWiseConv2dImplicit(
    in_channels=64,
    out_channels=64,
    big_kernel=51,      # 等效大核尺寸
    small_kernel=3,     # 实际小核尺寸
    ghost_ratio=0.23,   # Ghost通道比例
    N_path=2,          # 多路径数量
    N_rep=4            # 混洗排序组数
)

# 使用方式与标准Conv2d完全相同
x = torch.randn(2, 64, 56, 56)
y = conv(x)  # 输出形状与输入相同: (2, 64, 56, 56)
```

### 替换网络中的标准卷积

```python
import torch.nn as nn
from shiftwise_plug_and_play import ShifthWiseConv2dImplicit

# 原始网络
class OriginalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)

# 替换为Shift-Wise Conv
class ShiftWiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 第一层保持标准卷积
        self.conv2 = ShifthWiseConv2dImplicit(
            64, 128, big_kernel=51, small_kernel=3
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # 直接使用，无需修改forward逻辑
        return x
```

### 重参数化优化（训练后使用）

```python
# 训练完成后，可以合并多个路径以减少推理计算量
model.eval()
for module in model.modules():
    if isinstance(module, ShifthWiseConv2dImplicit):
        module.merge_branches()  # 合并N_path个并行卷积为一个
```

## 参数说明

- `in_channels`: 输入通道数
- `out_channels`: 输出通道数（通常等于in_channels）
- `big_kernel`: 等效大核尺寸（如51, 49, 47, 13等）
- `small_kernel`: 实际使用的小核尺寸（默认3）
- `stride`: 步长（默认1）
- `bn`: 是否使用BatchNorm（默认True）
- `ghost_ratio`: Ghost通道比例（默认0.23，即23%通道直接通过）
- `N_path`: 多路径数量（默认2，增加可提升特征利用率）
- `N_rep`: 混洗排序组数（默认4，控制特征多样性）
- `version`: "v1"（SLaK风格）或"v2"（UniRep风格，确保repN为偶数）

## 注意事项

1. **CUDA加速**: 如果安装了`shiftadd` CUDA扩展模块，会自动使用CUDA加速。否则使用CPU fallback实现（功能相同，但速度较慢）。

2. **兼容性**: 
   - 输入输出形状完全兼容标准Conv2d: `(B, C, H, W) -> (B, C, H, W)`
   - 可直接替换，无需修改网络其他部分

3. **性能**: 
   - 建议在有CUDA支持的环境中使用以获得最佳性能
   - 训练后使用`merge_branches()`可进一步优化推理速度

## 测试输出说明

运行测试脚本会执行以下测试：

1. **基础功能测试**: 验证输入输出形状、参数数量
2. **替换Conv2d测试**: 验证可以直接替换标准卷积层
3. **重参数化测试**: 验证merge_branches功能
4. **性能测试**: 测量推理时间和吞吐量

## 常见问题

**Q: 没有CUDA支持怎么办？**  
A: 模块会自动使用CPU fallback实现，功能完全相同，只是速度较慢。

**Q: 如何选择合适的big_kernel？**  
A: 根据任务需求选择，常用值有51, 49, 47, 13等。更大的值对应更大的感受野。

**Q: N_path和N_rep如何选择？**  
A: 
- N_path: 2-4较常用，增加会提升特征利用率但也会增加计算量
- N_rep: 4较常用，控制混洗排序的组数，影响特征多样性

