# MobileMamba 即插即用模块说明

本仓库提供了 `model/mobilemamba/mobilemamba_plug.py`，将 MobileMamba 中的核心混合模块抽离为独立组件，便于在任意 CNN/视觉骨干中复用。下列内容总结了模块构成、环境依赖以及基本测试/集成方法。

## 模块概览

- **MobileMambaModule**：将输入通道按比例划分为全局（SSM + 小波）、局部（深度可分卷积）和身份保留三支路，输出与输入形状一致，可直接堆叠或替换原有特征变换层。
- **MobileMambaBlock**：示例性残差块，展示如何将模块与前后 3×3 DWConv、1×1 FFN 组合。
- 依赖 SS2D（`model/lib_mamba/vmambanew.py`）。为适配无 Triton 的环境，`csm_triton.py` 与 `csm_tritonk2.py` 内置了简易 stub，可自动回退至 PyTorch 实现（性能略低）。

## 环境与依赖

在 Anaconda 的 `torchv5` 环境中验证，需确保安装：

- `pytorch` / `torchvision`
- `timm`
- `pywavelets`
- （可选）`triton`、`fvcore` —— 未安装时会自动回退，运行但速度降低

```bash
conda activate torchv5
pip install timm pywavelets fvcore  # 视环境而定
```

## 快速测试

`mobilemamba_plug.py` 自带 `test_mobilemamba_module()`，在 CPU 环境也可运行：

```bash
cd D:\upgithubfiles\deep-learning-papers-with-code\注意力机制\MobileMamba\MobileMamba-main\MobileMamba-main
conda activate torchv5
python model\mobilemamba\mobilemamba_plug.py
```

输出示例：

```
Input shape:  torch.Size([2, 192, 14, 14])
Output shape: torch.Size([2, 192, 14, 14])
```

同时可能看到关于 Triton/AMP 的警告，仅表示回退路径被启用，可忽略。

## 集成指引

1. 在目标工程中导入模块：
   ```python
   from model.mobilemamba.mobilemamba_plug import MobileMambaModule
   ```
2. 根据特征图通道数 `dim` 与期望的全局/局部比设置 `global_ratio`、`local_ratio`、`kernels` 等。
3. 将模块插入到现有 `(B,C,H,W)` 特征流中，输入输出分辨率保持一致，可配合残差连接或深度卷积使用。
4. 若在 GPU + Triton 环境运行，建议安装官方 Triton 包以恢复高性能扫描核。

## 常见问题

- **ModuleNotFoundError: model.lib_mamba**  
  运行脚本时请在仓库根目录执行，以便 `mobilemamba_plug.py` 内的路径注入逻辑生效。

- **Triton 未安装警告**  
  仅影响速度，不影响数值正确性。需要高性能时在支持的平台上安装 Triton。

- **Selective scan CUDA 警告**  
  表示 `csms6s` 的 CUDA 扩展未编译，同样会自动降级为 PyTorch 实现。

如需扩展或在其它项目中集成，可以此文件为起点，自行调整通道划分策略与下游头部。若有问题，欢迎根据需求进一步细化测试或补充文档。***

