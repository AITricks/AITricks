# PG-DRFNet 即插即用模块

本目录包含了从PG-DRFNet项目中提取的即插即用模块,这些模块可以独立使用或集成到其他目标检测项目中。

## 模块列表

### 1. PGHead (位置引导头)
- **文件**: `pg_head.py`
- **功能**: 从多尺度特征图中生成位置引导信号,用于小目标检测
- **特点**: 
  - 可配置的卷积层数量
  - 支持多尺度特征输入
  - 独立模块,不依赖框架特定组件

### 2. DynamicPerceptionV1 (动态感知 v1)
- **文件**: `dynamic_perception_v1.py`
- **功能**: 探索性版本的动态感知算法,使用堆叠方式构建特征
- **特点**:
  - 基于关键位置的探索性方法
  - 堆叠方式特征构建
  - 支持动态阈值筛选

### 3. DynamicPerceptionV2 (动态感知 v2)
- **文件**: `dynamic_perception_v2.py`
- **功能**: 优化版本的动态感知算法,使用扁平化方式构建特征
- **特点**:
  - 扁平化特征构建,具有更好的空间结构
  - 区域聚合优化
  - 相比v1版本更高效

## 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy
```

### 运行测试

```bash
python test_plugins.py
```

## 使用示例

### 使用 PGHead

```python
from pg_head import PGHead
import torch

# 创建模块
pg_head = PGHead(
    in_channels=192,
    conv_channels=192,
    num_convs=2,
    pred_channels=1
)

# 准备输入 (多尺度特征图)
features = [
    torch.randn(1, 192, 64, 64),
    torch.randn(1, 192, 32, 32),
]

# 前向传播
guidance_signals = pg_head(features)
```

### 使用 DynamicPerceptionV1

```python
from dynamic_perception_v1 import DynamicPerceptionV1
import torch

# 创建模块
dp = DynamicPerceptionV1(
    anchor_num=1,
    num_classes=18,
    score_th=0.12,
    context=2
)

# 准备输入
features_value = [
    torch.randn(1, 192, 64, 64),
    torch.randn(1, 192, 32, 32),
]

query_logits = [
    torch.randn(1, 1, 64, 64),
    torch.randn(1, 1, 32, 32),
    torch.randn(1, 1, 16, 16),
]

# 运行动态感知
block_features, inds = dp.run_dpinfer(features_value, query_logits)
```

### 集成使用

```python
from pg_head import PGHead
from dynamic_perception_v1 import DynamicPerceptionV1
import torch

# 1. 使用PGHead生成位置引导信号
pg_head = PGHead(192, 192, 2, 1)
features = [torch.randn(1, 192, 64, 64), torch.randn(1, 192, 32, 32)]
guidance_signals = pg_head(features)

# 2. 使用引导信号进行动态感知
dp = DynamicPerceptionV1(1, 18, 0.12, 2)
features_value = [torch.randn(1, 192, 64, 64), torch.randn(1, 192, 32, 32)]
query_logits = guidance_signals + [torch.randn(1, 1, 16, 16)]

block_features, inds = dp.run_dpinfer(features_value, query_logits)
```

## 模块特性

### PGHead
- **输入**: 多尺度特征图列表 `List[Tensor]`, 每个形状为 `[B, C, H, W]`
- **输出**: 位置引导信号列表 `List[Tensor]`, 每个形状为 `[B, pred_channels, H, W]`
- **可配置参数**:
  - `in_channels`: 输入特征通道数
  - `conv_channels`: 中间卷积层通道数
  - `num_convs`: 卷积层数量
  - `pred_channels`: 预测输出通道数 (通常为1)

### DynamicPerceptionV1/V2
- **输入**: 
  - `features_value`: 多尺度特征值列表
  - `query_logits`: 查询逻辑值列表 (需要比features_value多一个元素)
- **输出**: 
  - `block_feature_list`: 提取的特征块列表
  - `inds`: 索引信息
- **可配置参数**:
  - `anchor_num`: 锚框数量
  - `num_classes`: 类别数量
  - `score_th`: 分数阈值 (默认0.12)
  - `context`: 上下文范围 (默认2或4)

## 在项目中的位置

这些模块在原始PG-DRFNet项目中的位置:
- **PGHead**: `mmrotate/models/dense_heads/rotated_combination_head.py` (第1079-1112行)
- **DynamicPerceptionV1**: `utils/dpinfer_v1.py`
- **DynamicPerceptionV2**: `utils/dpinfer_v2.py`

## 架构说明

根据PG-DRFNet的架构图:

1. **Positional Guidance Head (PGH)**: 接收多级特征和OBB中心,生成位置引导信号
2. **Dynamic Perception Head (DPH)**: 使用关键位置网格,通过引导生成响应区域,提取特征
3. **Dynamic Perception Algorithm**: 
   - V1 (探索性版本): 使用堆叠方式构建特征
   - V2 (优化版本): 使用扁平化方式构建特征,具有更好的空间结构

## 注意事项

1. 这些模块已经去除对mmcv和mmdet框架的依赖,可以独立使用
2. 确保输入的张量维度匹配
3. 动态感知模块要求batch_size=1
4. 根据具体任务调整`score_th`和`context`参数

## 参考文献

Position Guided Dynamic Receptive Field Network: A Small Object Detection Friendly to Optical and SAR Images

