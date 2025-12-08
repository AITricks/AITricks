# MFF-Net 即插即用模块说明

本文档对应论文图 2~4 中的可复用模块，实现位置在 `MFF-Net/plug_and_play_modules.py`。

## 模块列表

1. **CrossAttention**  
   - 来源：图 3 中的 Cross Attention。  
   - 输入：查询特征 `fea`（形状 `[B, N1, dim1]`）和键值特征 `aux_fea`（形状 `[B, N2, dim2]`）。  
   - 功能：将不同尺度/分支的特征进行跨注意力融合，可直接用于多模态或多尺度交互。

2. **DirectionalConvUnit**  
   - 来源：图 2 中 MSFU 模块前的方向卷积单元。  
   - 功能：对特征分别施加水平、垂直以及两条对角线方向的卷积，再拼接输出，强化结构信息。

3. **SWSAM（Sliding Window Spatial Attention Module）**  
   - 来源：图 2 中的 SWSAM。  
   - 功能：先做通道洗牌再划分为 4 组，分别做空间注意力并使用可学习权重融合，最后通过 1×1 Conv 进行细化，实现轻量的局部注意力增强。

模块中还包含以下辅助组件：

- `SpatialAttention`：单分支空间注意力计算。  
- `BasicConv2d`：带 BN 的简单卷积层。  
- `channel_shuffle`：通道洗牌操作。

## 使用方式

```python
from plug_and_play_modules import CrossAttention, DirectionalConvUnit, SWSAM
```

- 将模块直接实例化并嵌入任意网络结构即可，输入输出张量形状同文件内注释。
- 这些模块依赖 `torch`, `torch.nn.functional`，不再依赖项目其他脚本，可独立复用。

## 快速测试

文件末尾提供了 `if __name__ == "__main__":` 入口，运行：

```bash
python MFF-Net/plug_and_play_modules.py
```

脚本会自动：

1. 构造随机张量；
2. 在 GPU（若可用）或 CPU 上执行 SWSAM、DirectionalConvUnit、CrossAttention；
3. 打印各模块输出的张量形状，确认可用性。

如需自定义测试，只需改写 `main` 中的输入尺寸或链路即可。

