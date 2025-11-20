# TDCNet 即插即用模块说明

为便于在其他工程中复用 TDCNet 的核心部件，我们将模型结构图（图1–4）中的模块抽取到 `tdc_plug_modules.py`。本文件简要说明各模块与原论文结构图的对应关系以及使用方法。

## 模块一览

| 模块 | 对应图 | 说明 |
| --- | --- | --- |
| `TDC` | 图4(b) | 计算时间差分卷积，支持短/中/长期步长。 |
| `RepConv3D` | 图3 | 将三路 TDC (S/M/L) 重参数化为单个 3D 卷积核，可在部署时切换为 `deploy` 模式。 |
| `WindowAttention3D` | 图2 | 3D 窗口注意力的基础实现，支持相对位置编码。 |
| `SelfAttention` | 图2（TDC-guided SA） | 两阶段窗口自注意力 + MLP，支持窗口平移（shift）。 |
| `CrossAttention` | 图2（TDC-guided CA） | 以 3D 特征为 Query、Key/Value 互补的窗口跨注意力。 |

> 图1 仅展示整体流程对比，没有独立的可拆分模块。

## 使用方式

1. **导入模块**  
   ```python
   from model.TDCNet.tdc_plug_modules import TDC, RepConv3D, SelfAttention, CrossAttention
   ```

2. **作为 3D 主干替换块**  
   ```python
   block = RepConv3D(in_channels=64, out_channels=128, stride=(1,2,2))
   feat = block(video_tensor)
   ```

3. **在 Transformer 中复用注意力**  
   ```python
   sa = SelfAttention(dim=256, window_size=(2,4,4), num_heads=4, use_shift=True)
   out = sa(spatiotemporal_feat)  # [B, T, H, W, C]
   ```

4. **跨模态/跨分支交互**  
   ```python
   ca = CrossAttention(dim=128, window_size=(2,8,8), num_heads=4)
   fused = ca(q_feat, k_feat, v_feat)
   ```

## 注意事项

- `SelfAttention`/`CrossAttention` 期望输入形状为 `[B, T, H, W, C]`，请先 `permute` 并确保窗口大小与特征尺寸匹配。
- `RepConv3D` 在训练阶段由三路 TDC 组成；调用 `switch_to_deploy()` 后会删除分支并创建单个 3D 卷积，适合推理部署。
- `TDC` 的 `step` 控制时间差分跨度，可根据任务定制短/中/长期信息。

如需整个 `TDCNetwork` 架构用于迁移，可直接引用 `model/TDCNet/TDCNetwork.py`，但它包含完整检测流水线，不属于轻量组件。

