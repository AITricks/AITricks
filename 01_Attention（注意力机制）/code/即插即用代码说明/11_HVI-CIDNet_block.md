# plug_and_play_cidnet.py 使用说明

本文档介绍如何在单个 `net/plug_and_play_cidnet.py` 文件中使用 CIDNet 的全部即插即用模块，并给出运行示例。

## 模块概览
- **Transformer Utilities**：`LayerNorm`、`NormDownsample`、`NormUpsample`，实现归一化与上/下采样结构。
- **Lighten Cross-Attention**：`CAB`、`IEL`、`HV_LCA`、`I_LCA`，对应图中轻量交叉注意力与强度增强分支。
- **HVI 颜色空间**：`RGB_HVI` 提供 HVIT/PHVIT 前后变换。
- **CIDNet 主体**：`CIDNet` 将上述组件组合成双分支 U-Net 结构，可直接做低照度增强。

所有类与函数均在同一个文件中定义，无需引用仓库其他模块，便于复制到新项目。

## 依赖
- Python 3.8+
- `torch`（建议与项目训练环境一致）
- `einops`

示例安装：
```bash
pip install torch einops
```

## 使用方法
1. 保持当前目录在仓库根目录或包含 `net` 目录的位置。
2. 运行烟雾测试：
   ```bash
   python -m net.plug_and_play_cidnet
   ```
   程序会自动选择 CPU/CUDA，随机生成 `1×3×128×128` 的输入图像，通过模型前向传播，并打印输入/输出张量形状。
3. 在其他项目中，只需复制 `net/plug_and_play_cidnet.py`，再按需导入其中的类或直接构建 `CIDNet`：
   ```python
   from plug_and_play_cidnet import build_cidnet

   model = build_cidnet()
   enhanced = model(low_light_img)
   ```

## 可选配置
- `build_cidnet(channels=..., heads=..., norm=False)`：可调整各层通道数、注意力头数、是否开启 LayerNorm。
- `RGB_HVI` 中的 `density_k`、`gated` 等参数也可手动修改以匹配特定数据集。

## 测试建议
复制文件到新环境后，先运行 `python plug_and_play_cidnet.py`（或引用 `main()`）确保依赖正确，再集成到实际训练/推理脚本。

