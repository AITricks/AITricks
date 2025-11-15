# AMD 即插即用模块说明文档

## 概述

本文档介绍了 AMD (Adaptive Multi-predictor Decomposition) 架构中的即插即用模块。这些模块可以独立使用或组合使用，用于时间序列预测任务。

## 模块架构

根据 AMD 架构图，整个系统包含三个主要阶段：

1. **Multi-Scale Decomposable Mixing (MDM)** - 多尺度可分解混合
2. **Dual Dependency Interaction (DDI)** - 双依赖交互
3. **Adaptive Multi-predictor Synthesis (AMS)** - 自适应多预测器合成

## 模块列表

### 1. RevIN - 可逆实例归一化

**功能**: 对时间序列进行可逆的实例归一化，提高模型的泛化能力。

**特点**:
- 支持归一化和反归一化两种模式
- 可学习的仿射参数
- 保持时间序列的统计特性

**使用示例**:
```python
from plug_and_play_modules import RevIN

# 初始化
revin = RevIN(num_features=7)

# 归一化
x_norm = revin(x, mode='norm')

# 反归一化
x_denorm = revin(x_norm, mode='denorm', target_slice=slice(None))
```

**参数**:
- `num_features`: 特征或通道数
- `eps`: 数值稳定性参数（默认: 1e-5）
- `affine`: 是否使用可学习的仿射参数（默认: True）

**输入/输出形状**:
- 输入: `[batch_size, seq_len, num_features]`
- 输出: `[batch_size, seq_len, num_features]`

---

### 2. MDM - 多尺度可分解混合

**功能**: 将输入分解为多个时间尺度并进行混合，捕获不同时间尺度的信息。

**特点**:
- 多尺度下采样和混合
- 可配置的尺度层数和缩放因子
- 可选层归一化

**使用示例**:
```python
from plug_and_play_modules import MDM

# 初始化
mdm = MDM(input_shape=(96, 7), k=3, c=2, layernorm=True)

# 前向传播
# 注意：输入需要是 [batch, feature, seq_len] 格式
x_mdm = torch.randn(4, 7, 96)
output = mdm(x_mdm)
```

**参数**:
- `input_shape`: 输入形状 `[seq_len, feature_num]`
- `k`: 多尺度层数（默认: 3）
- `c`: 尺度缩放因子（默认: 2）
- `layernorm`: 是否使用层归一化（默认: True）

**输入/输出形状**:
- 输入: `[batch_size, feature_num, seq_len]`
- 输出: `[batch_size, feature_num, seq_len]`

---

### 3. DDI - 双依赖交互

**功能**: 建模不同尺度之间的动态交互关系，通过 patch 机制处理长序列。

**特点**:
- Patch-based 处理机制
- 可配置的特征交互权重
- 支持历史信息聚合

**使用示例**:
```python
from plug_and_play_modules import DDI

# 初始化
ddi = DDI(input_shape=(96, 7), dropout=0.1, patch=12, alpha=0.5, layernorm=True)

# 前向传播
x_ddi = torch.randn(4, 7, 96)
output = ddi(x_ddi)
```

**参数**:
- `input_shape`: 输入形状 `[seq_len, feature_num]`
- `dropout`: Dropout 率（默认: 0.2）
- `patch`: Patch 大小（默认: 12）
- `alpha`: 特征交互权重（默认: 0.0）
- `layernorm`: 是否使用层归一化（默认: True）

**输入/输出形状**:
- 输入: `[batch_size, feature_num, seq_len]`
- 输出: `[batch_size, feature_num, seq_len]`

---

### 4. TopKGating - Top-K 门控机制

**功能**: 选择最重要的专家进行预测，实现稀疏激活。

**特点**:
- Top-K 专家选择
- 训练时的噪声注入机制
- 可学习的门控权重

**使用示例**:
```python
from plug_and_play_modules import TopKGating

# 初始化
gating = TopKGating(input_dim=96, num_experts=4, top_k=2)

# 前向传播
x_gating = torch.randn(4, 96)
gates = gating(x_gating)  # [batch_size, num_experts]
```

**参数**:
- `input_dim`: 输入维度
- `num_experts`: 专家数量
- `top_k`: 选择的 top-k 专家数（默认: 2）
- `noise_epsilon`: 噪声 epsilon（默认: 1e-5）

**输入/输出形状**:
- 输入: `[batch_size, input_dim]`
- 输出: `[batch_size, num_experts]` (门控权重，每行和为1)

---

### 5. Expert - 专家网络

**功能**: 单个预测器，用于处理特定的时间模式。

**特点**:
- 简单的 MLP 结构
- 可配置的隐藏层维度
- Dropout 正则化

**使用示例**:
```python
from plug_and_play_modules import Expert

# 初始化
expert = Expert(input_dim=96, output_dim=24, hidden_dim=512, dropout=0.1)

# 前向传播
x_expert = torch.randn(4, 96)
output = expert(x_expert)  # [batch_size, 24]
```

**参数**:
- `input_dim`: 输入维度
- `output_dim`: 输出维度（预测长度）
- `hidden_dim`: 隐藏层维度
- `dropout`: Dropout 率（默认: 0.2）

**输入/输出形状**:
- 输入: `[batch_size, input_dim]`
- 输出: `[batch_size, output_dim]`

---

### 6. AMS - 自适应多预测器合成

**功能**: 根据时间模式自适应选择并组合多个预测器，实现多专家混合（MoE）。

**特点**:
- 多专家集成
- 自适应权重分配
- 负载均衡损失

**使用示例**:
```python
from plug_and_play_modules import AMS

# 初始化
ams = AMS(input_shape=(96, 7), pred_len=24, 
          ff_dim=512, dropout=0.1, num_experts=4, top_k=2)

# 前向传播
x_ams = torch.randn(4, 7, 96)
time_emb = torch.randn(4, 7, 96)
pred, moe_loss = ams(x_ams, time_emb)
```

**参数**:
- `input_shape`: 输入形状 `[seq_len, feature_num]`
- `pred_len`: 预测长度
- `ff_dim`: 前馈网络维度（默认: 2048）
- `dropout`: Dropout 率（默认: 0.2）
- `loss_coef`: 损失系数（默认: 1.0）
- `num_experts`: 专家数量（默认: 4）
- `top_k`: Top-k 专家数（默认: 2）

**输入/输出形状**:
- 输入: 
  - `x`: `[batch_size, feature_num, seq_len]`
  - `time_embedding`: `[batch_size, feature_num, seq_len]`
- 输出:
  - `pred`: `[batch_size, feature_num, pred_len]`
  - `moe_loss`: 标量（负载均衡损失）

---

## 完整使用流程

以下是一个完整的使用示例，展示如何组合这些模块：

```python
import torch
from plug_and_play_modules import RevIN, MDM, DDI, AMS

# 初始化参数
batch_size = 4
seq_len = 96
pred_len = 24
feature_num = 7

# 初始化模块
revin = RevIN(num_features=feature_num)
mdm = MDM(input_shape=(seq_len, feature_num), k=3, c=2, layernorm=True)
ddi = DDI(input_shape=(seq_len, feature_num), dropout=0.1, patch=12, alpha=0.5, layernorm=True)
ams = AMS(input_shape=(seq_len, feature_num), pred_len=pred_len, 
          ff_dim=512, dropout=0.1, num_experts=4, top_k=2)

# 准备输入数据 [batch_size, seq_len, feature_num]
x = torch.randn(batch_size, seq_len, feature_num)

# 1. RevIN 归一化
x = revin(x, mode='norm')

# 2. 转置为 [batch_size, feature_num, seq_len]
x = x.transpose(1, 2)

# 3. MDM 多尺度混合（生成时间嵌入）
time_embedding = mdm(x)

# 4. DDI 双依赖交互
x = ddi(x)

# 5. AMS 自适应多预测器合成
pred, moe_loss = ams(x, time_embedding)

# 6. 转回 [batch_size, pred_len, feature_num]
pred = pred.transpose(1, 2)

# 7. RevIN 反归一化
pred = revin(pred, mode='denorm', target_slice=slice(None))

print(f"预测结果形状: {pred.shape}")  # [4, 24, 7]
print(f"MoE损失: {moe_loss.item():.6f}")
```

## 测试

运行测试函数验证所有模块的功能：

```bash
python plug_and_play_modules.py
```

测试包括：
- ✅ 各模块独立功能测试
- ✅ 输入输出形状验证
- ✅ 模块组合流程测试
- ✅ 数值稳定性检查

## 模块特性总结

| 模块 | 主要功能 | 是否可独立使用 | 输入格式 |
|------|---------|---------------|---------|
| RevIN | 归一化/反归一化 | ✅ | `[B, L, F]` |
| MDM | 多尺度混合 | ✅ | `[B, F, L]` |
| DDI | 双依赖交互 | ✅ | `[B, F, L]` |
| TopKGating | 专家选择 | ✅ | `[B, L]` |
| Expert | 单预测器 | ✅ | `[B, L]` |
| AMS | 多预测器合成 | ✅ | `[B, F, L]` |

**说明**:
- `B`: batch_size
- `L`: seq_len (序列长度)
- `F`: feature_num (特征数)

## 注意事项

1. **数据格式**: 注意不同模块期望的输入格式（`[B, L, F]` vs `[B, F, L]`），需要适当转置。

2. **设备一致性**: 确保所有模块和输入数据在同一设备上（CPU 或 GPU）。

3. **训练/推理模式**: 
   - `TopKGating` 在训练时会注入噪声，推理时使用干净的门控值
   - 使用 `model.train()` 和 `model.eval()` 切换模式

4. **序列长度**: 确保 `seq_len` 能被 `patch` 整除（DDI 模块要求）。

5. **内存使用**: 
   - `AMS` 模块会创建多个专家网络，注意内存占用
   - 可以通过调整 `num_experts` 和 `ff_dim` 来控制模型大小

## 扩展使用

这些模块可以灵活组合，例如：

- **仅使用 MDM**: 用于多尺度特征提取
- **仅使用 AMS**: 用于多专家预测
- **组合 MDM + DDI**: 用于多尺度交互建模
- **组合所有模块**: 完整的 AMD 架构

## 参考文献

基于 AMD (Adaptive Multi-predictor Decomposition) 架构设计，用于时间序列预测任务。

## 许可证

请参考项目根目录的 LICENSE 文件。

