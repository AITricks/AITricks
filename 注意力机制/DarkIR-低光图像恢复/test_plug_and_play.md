### DarkIR 即插即用模块说明（EBlock / DBlock）

本文件介绍 `test_plug_and_play.py` 中的轻量即插即用增强与解码模块，便于在任意 PyTorch 网络中快速复用与对比。

---

## 模块概览
- **LayerNorm2d**: 通道维度归一化，等价于对每个样本的每个通道按 H×W 统计量做 LayerNorm。
- **FreMLP**: 频域幅度调制模块。将输入做 FFT，学习幅度（magnitude）后重建复数谱，再 iFFT 回像素域，实现基于频率的光照/对比增强。
- **EBlock (Encoder Block)**: 编码端块，包含深度可分组多分支空洞卷积、Squeeze-Channel Attention、频域调制（FreMLP），适合低照与退化信息提取。
- **DBlock (Decoder Block)**: 解码端块，结构与 EBlock 相似，但加入门控前馈（Gated-FFN）以提升恢复/重建能力；支持可选的额外深度卷积与多空洞率并行分支。

---

## 依赖与环境
- Python 3.8+
- PyTorch 1.10+
- CUDA 可选（自动检测）

项目根目录的 `requirements.txt` 已涵盖主要依赖。

---

## 张量约定
- 输入/输出形状: `(B, C, H, W)`
- 数据类型: `torch.float32`

---

## API 说明

### LayerNorm2d(channels: int, eps: float = 1e-6)
- **作用**: 对每个样本的每个通道在空间维上做归一化。
- **参数**:
  - `channels`: 通道数 C。
  - `eps`: 数值稳定项。

### FreMLP(nc: int, expand: int = 2)
- **作用**: 在频域学习幅度响应，用以调节照明/纹理分布。
- **参数**:
  - `nc`: 通道数。
  - `expand`: 中间通道扩张倍率（1×1 卷积）。

### EBlock(c: int, DW_Expand: int = 2, dilations: List[int] = [1], extra_depth_wise: bool = False)
- **作用**: 编码块，聚合多尺度上下文并进行频域调制。
- **参数**:
  - `c`: 输入/输出通道数。
  - `DW_Expand`: 深度可分支前的通道扩张倍率（决定内部通道 `DW_Expand*c`）。
  - `dilations`: 多分支空洞卷积的空洞率列表，如 `[1, 3, 5]`。
  - `extra_depth_wise`: 是否在 1×1 前增加一层 3×3 深度卷积以加强局部建模。
- **输入/输出**: `x -> (B,C,H,W)`，返回同形状张量。

### DBlock(c: int, DW_Expand: int = 2, FFN_Expand: int = 2, dilations: List[int] = [1], extra_depth_wise: bool = False)
- **作用**: 解码块，包含多分支空洞卷积 + 通道注意力 + 门控前馈（Gated-FFN）。
- **参数**:
  - `c`: 输入/输出通道数。
  - `DW_Expand`: 同上。
  - `FFN_Expand`: FFN 的通道扩张倍率。
  - `dilations`: 空洞率列表，如 `[1, 3]`。
  - `extra_depth_wise`: 同上。
- **输入/输出**: `x -> (B,C,H,W)`，返回同形状张量。

---

## 快速上手
在根目录下运行：
```bash
python test_plug_and_play.py
```
将打印 `EBlock` 与 `DBlock` 的输入/输出形状，验证可独立前向推理。

在你自己的网络中使用：
```python
from test_plug_and_play import EBlock, DBlock
import torch

x = torch.randn(1, 32, 64, 64).cuda()  # 或 .cpu()

# 编码端示例：多尺度空洞 + 频域调制
encoder_block = EBlock(32, DW_Expand=2, dilations=[1, 3, 5], extra_depth_wise=True)
y_enc = encoder_block(x)

# 解码端示例：多尺度空洞 + Gated-FFN
decoder_block = DBlock(32, DW_Expand=2, FFN_Expand=2, dilations=[1, 3], extra_depth_wise=True)
y_dec = decoder_block(x)
```

---

## 设计细节与建议
- **多分支空洞卷积**: 通过并行不同空洞率扩大感受野，适合噪声/模糊/低照场景的上下文聚合。
- **Squeeze-Channel Attention**: 使用 `AdaptiveAvgPool2d(1)` + `1×1` 卷积建模通道间依赖；配合 `SimpleGate`（通道二分相乘）提升非线性表达。
- **频域调制（EBlock）**: 对 `norm→conv` 后的特征做 `rFFT→幅度学习→iFFT`，再与残差融合以抑制伪影。
- **门控前馈（DBlock）**: `FFN_Expand` 控制 FFN 维度，`SimpleGate` 实现轻量化门控。
- **数值稳定**: 归一化采用 `eps=1e-6`；频域重建保持 `norm='backward'` 与输入能量一致性。

---

## 常见问题
- **是否可更换注意力/激活?** 可以。`SimpleGate` 和通道注意力均为可替换组件。
- **与上采样/下采样如何衔接?** 两个模块均保持分辨率不变；与外部 `ConvTranspose2d`、`PixelShuffle` 或下采样层级联即可。
- **推理设备**: 自动检测 CUDA；若显存受限，可减小 `DW_Expand/FFN_Expand` 或删减 `dilations` 分支。

---

## 最小复现实验
`test_plug_and_play.py` 的 `__main__` 已内置：
- 生成随机输入 `(1, 32, 64, 64)`；
- 构建 `EBlock(dilations=[1,3,5])` 与 `DBlock(dilations=[1,3])`；
- 打印前后张量形状用于快速自检。

---

## 许可
遵循项目根目录的 `LICENSE`。


