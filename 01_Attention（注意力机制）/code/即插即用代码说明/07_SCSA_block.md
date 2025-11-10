## SCSAPlug 即插即用模块说明

`scsa_plug.py` 提供了一个不依赖 mmengine/mmpretrain 的纯 PyTorch SCSA 注意力模块，可直接插入任意 CNN 模块之间使用。输入/输出均为 `(B, C, H, W)`，与结构图对应包含两部分：
- Shared Multi-Semantic Spatial Attention（SMSA）：沿 H/W 的多尺度 1D 深度可分组卷积 + GroupNorm(4) + gate；
- Progressive Channel-wise Self-Attention（PCSA）：池化降采样 → 通道自注意（支持多头）→ 上采样复原。

### 文件
- 模块与示例：`scsa_plug.py`
  - 类：`SCSAPlug(nn.Module)`
  - 自带最小化 `main()`，可直接跑一次前向测试

### 环境
- Python 3.x
- PyTorch（与你的 Anaconda `torchv5` 环境兼容）

### 运行示例
在项目根目录执行：

```bash
python scsa_plug.py
```

若有 GPU 会自动使用 CUDA；否则使用 CPU。输出会打印输入与输出张量的形状。

### API
```python
from scsa_plug import SCSAPlug
import torch

model = SCSAPlug(
	dim=64,             # 输入通道 C，需能被 4 和 head_num 整除
	head_num=1,         # 通道自注意的 head 数
	window_size=7,      # PCSA 降采样窗口（控制计算量）
	group_kernel_sizes=[3, 5, 7, 9],  # SMSA 多尺度 1D 深度卷积核
	down_sample_mode="avg",           # "avg" or "max"
	gate="sigmoid",                   # "sigmoid" or "tanh"
)

x = torch.randn(2, 64, 56, 56)
y = model(x)  # y.shape == x.shape
```

参数约束：
- `dim % 4 == 0`（SMSA 四分组）；
- `dim % head_num == 0`（多头通道注意）。

### 集成到你的网络
以一个卷积块为例：
```python
class MyBlock(torch.nn.Module):
	def __init__(self, channels: int):
		super().__init__()
		self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(channels)
		self.act = torch.nn.ReLU(inplace=True)
		self.attn = SCSAPlug(dim=channels, head_num=1, window_size=7)

	def forward(self, x):
		x = self.act(self.bn1(self.conv1(x)))
		x = self.attn(x)   # 即插即用
		return x
```

### 设计对应关系（与结构图）
- SMSA：对 H/W 做 `mean` 聚合，四等分通道后分别用不同核长的 1D 深度卷积，再经 GroupNorm(4) 与门控（Sigmoid/Tanh）得到纵横注意图，外积作用于输入特征。
- PCSA：按 `window_size` 池化降采样后进行通道自注意（q/k/v 为 1×1 卷积，多头按通道分组），最后上采样回原尺寸并与 SMSA 输出相加。

### 故障排查
- 报错 “dim must be divisible by 4”：请把 `dim` 设为 4 的倍数；
- 报错 “dim must be divisible by head_num”：请调整 `head_num` 使 `dim % head_num == 0`；
- 运行很慢：适当调大 `window_size`（降低 PCSA 计算量），或在 CPU 上改为更小的输入尺寸测试。


