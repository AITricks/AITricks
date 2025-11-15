## ARConv 模块说明

### 模块概览

`models/arconv_module.py` 提供了一个可即插即用的自适应感受野卷积层 `ARConv`。该层通过学习偏移量与调制参数，同时包含多组可变形卷积核，从而在不同输入尺度之间自适应调节感受野。

- 仅依赖 `torch` 与 `torch.nn`
- 前向接口：`forward(x, epoch, hw_range)`
- 通过 `_demo_forward()` 提供最小化示例，方便快速验证

### 参数说明

| 参数            | 类型 | 说明 |
|-----------------|------|------|
| `inc`           | int  | 输入通道数 |
| `outc`          | int  | 输出通道数 |
| `kernel_size`   | int  | 基础卷积核尺寸，默认 3 |
| `padding`       | int  | 零填充大小，默认 1 |
| `stride`        | int  | 步长，默认 1 |
| `l_max / w_max` | int  | 感受野搜索的最大长宽，默认 9 |
| `flag`          | bool | 预留开关，默认 False |
| `modulation`    | bool | 是否启用调制，默认 True |

`forward` 额外需要输入：

- `x`: 形状为 `(batch, inc, H, W)` 的输入特征
- `epoch`: 当前训练轮数（用于感受野自适应策略）
- `hw_range`: `[h_max, w_max]`，定义感受野范围

### 使用方式

```python
from models.arconv_module import ARConv

layer = ARConv(inc=3, outc=64)
output = layer(x, epoch=current_epoch, hw_range=[1, 9])
```

若项目仍引用 `models/ARConv.py`，该文件已经改为简单 re-export，可直接 `from models.ARConv import ARConv`。

### 快速测试

1. 在 Anaconda 中激活 `torchv5` 环境：

   ```bash
   conda activate torchv5
   ```

2. 运行内置示例，验证前向传播：

   ```bash
   python models/arconv_module.py
   ```

   输出将打印输入、输出张量形状，用于确认模块能独立工作。

### 集成建议

- 将 `ARConv` 替换或插入到现有 CNN 模型中之前，请确认下游层的输入尺寸兼容。
- 训练时 `epoch` 参数需随训练迭代更新，否则感受野可能无法正确冻结。
- 若在推理阶段使用，请确保在训练结束时 `remove_hooks()` 被调用，释放注册的 backward hooks（可选）。


