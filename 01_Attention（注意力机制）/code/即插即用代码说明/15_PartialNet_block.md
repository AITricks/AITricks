# PAT 即插即用模块说明

本文件说明 `models/pat_modules.py` 中封装的 PAT 模块，便于在 PartialNet 之外直接复用。

## 模块概览
- **`Partial_conv3`**：核心的部分通道卷积。  
  - `channel_type='se'` 时启用 **PAT_ch**（风格池化 + SRM 通道注意）。  
  - `channel_type='self'` 时启用 **PAT_sf**（RPE 注意力）。
- **`partial_spatial_attn_layer_reverse`**：实现 **PAT_sp**，对空间特征进行部分注意力重加权。
- **`RPEAttention` / `SRM`**：分别为 PAT_sf、PAT_ch 中的注意力子模块，可单独引用。

这些组件与 `models/partialnet.py` 内的实现完全一致，但独立于主 Backbone，方便外部网络按需组合。

## 依赖
```text
torch >= 1.10
timm
```
`RPEAttention` 依赖 `models/irpe.py` 中的相对位置编码构建函数，确保同目录可被正确 import。

## 快速上手
```python
from models.pat_modules import Partial_conv3, partial_spatial_attn_layer_reverse

pat_ch = Partial_conv3(dim=64, n_div=4, forward_type='split_cat',
                       use_attn=True, channel_type='se')
pat_sp = partial_spatial_attn_layer_reverse(dim=64, n_head=1, partial=0.5)
```
所有模块输入输出 shape 均与 PartialNet 中一致：`(B, C, H, W)`。

## 自带测试
`pat_modules.py` 内置 `main()`，会构造随机张量依次通过 PAT_ch、PAT_sf、PAT_sp：
```bash
python -m models.pat_modules
```
需要先安装 PyTorch。运行成功后会在控制台输出三个模块的输出尺寸，用于 sanity check。

## 与论文结构图的对应关系
- PAT_ch block → `Partial_conv3(channel_type='se')`
- PAT_sp block → `partial_spatial_attn_layer_reverse`
- PAT_sf block → `Partial_conv3(channel_type='self')` + `RPEAttention`

因此可在图中标记的任意位置将这些模块作为“即插即用”构件，直接 import 使用即可。

