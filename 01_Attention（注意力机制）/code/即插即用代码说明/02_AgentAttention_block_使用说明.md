# Agent Attention 即插即用模块使用说明

## 📖 模块简介

Agent Attention 是一个用于 Stable Diffusion 模型的即插即用加速模块，通过结合 **ToMe (Token Merging)** 和 **Agent Attention** 机制，可以在不重新训练模型的情况下：

- ✅ **加速推理速度**：减少计算量，提升生成速度
- ✅ **降低内存占用**：减少 GPU 内存使用
- ✅ **提升图像质量**：在加速的同时提升生成图像的质量
- ✅ **无需重训练**：直接应用到现有模型，无需额外训练

## 🚀 快速开始

### 1. 环境要求

```bash
# 必需的依赖包
torch >= 1.12.1
einops
```

### 2. 安装

将 `AgentAttention_block.py` 文件放置在您的项目目录中，确保 `agentsd` 目录（包含 `merge.py` 和 `utils.py`）在同一目录下。

```
项目目录/
├── AgentAttention_block.py
├── agentsd/
│   ├── merge.py
│   ├── utils.py
│   └── __init__.py
└── 您的代码.py
```

### 3. 测试模块

运行测试脚本验证模块是否正常工作：

```bash
python AgentAttention_block.py
```

如果看到所有测试通过，说明模块已正确安装。

## 💡 基本使用

### 方法一：直接导入使用

```python
from AgentAttention_block import apply_patch, remove_patch

# 加载您的 Stable Diffusion 模型
# model = ... 您的模型加载代码

# 应用 Agent Attention 补丁
apply_patch(
    model,
    ratio=0.4,           # token合并比例
    agent_ratio=0.8,     # agent token比例
    k_scale2=0.3,        # 第二阶段注意力缩放因子
    k_shortcut=0.075     # 残差连接系数
)

# 使用模型进行推理
# ... 您的推理代码 ...

# 可选：移除补丁，恢复原始模型
remove_patch(model)
```

### 方法二：在扩散过程中动态应用

```python
from AgentAttention_block import apply_patch, remove_patch

# 在扩散过程的不同阶段使用不同参数
for step in range(num_steps):
    if step == 0:
        # 早期步骤：使用较强的token合并
        apply_patch(model, ratio=0.4, agent_ratio=0.95, sx=4, sy=4)
    elif step == 20:
        # 后期步骤：使用较弱的token合并
        remove_patch(model)
        apply_patch(model, ratio=0.4, agent_ratio=0.5, sx=2, sy=2)
    
    # 执行扩散步骤
    # ... 您的扩散代码 ...
```

## 📚 详细参数说明

### `apply_patch()` 函数参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `torch.nn.Module` | - | Stable Diffusion 模型对象（必需） |
| `ratio` | `float` | `0.5` | Token合并比例，范围 [0, 1]，值越大合并越多 |
| `max_downsample` | `int` | `1` | 应用补丁的最大下采样层数（1, 2, 4, 8） |
| `sx`, `sy` | `int` | `2` | Token合并的stride（步长） |
| `agent_ratio` | `float` | `0.8` | Agent token生成时的合并比例 |
| `k_scale2` | `float` | `0.3` | Agent Attention第二阶段注意力的缩放因子 |
| `k_shortcut` | `float` | `0.075` | 残差连接系数 |
| `attn_precision` | `str` | `None` | 注意力计算精度，`"fp32"` 可避免数值不稳定（SD v2.1推荐） |
| `use_rand` | `bool` | `True` | 是否使用随机扰动 |
| `merge_attn` | `bool` | `True` | 是否在自注意力层合并tokens（推荐） |
| `merge_crossattn` | `bool` | `False` | 是否在交叉注意力层合并tokens（不推荐） |
| `merge_mlp` | `bool` | `False` | 是否在MLP层合并tokens（不推荐） |

### `remove_patch()` 函数

```python
remove_patch(model)
```

移除 Agent Attention 补丁，恢复原始模型。会清除所有hooks并将模块类恢复为原始类。

## 🎯 推荐配置

### Stable Diffusion v1.5

```python
apply_patch(
    model,
    ratio=0.4,
    agent_ratio=0.8,
    k_scale2=0.3,
    k_shortcut=0.075,
    max_downsample=1,
    sx=2,
    sy=2
)
```

### Stable Diffusion v2.1

```python
apply_patch(
    model,
    ratio=0.4,
    agent_ratio=0.8,
    k_scale2=0.3,
    k_shortcut=0.075,
    attn_precision="fp32",  # 重要：避免数值不稳定
    max_downsample=1,
    sx=2,
    sy=2
)
```

### 高分辨率生成（512x512及以上）

```python
apply_patch(
    model,
    ratio=0.3,          # 降低合并比例以保持质量
    agent_ratio=0.7,
    max_downsample=2,   # 允许更多层应用补丁
    sx=4,
    sy=4
)
```

## 📝 完整示例

### 示例1：基本使用（Stable Diffusion v1.5）

```python
import torch
from diffusers import StableDiffusionPipeline
from AgentAttention_block import apply_patch, remove_patch

# 加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to(device)

# 应用 Agent Attention 补丁
apply_patch(pipe, ratio=0.4, agent_ratio=0.8)

# 生成图像
prompt = "a beautiful landscape"
image = pipe(prompt).images[0]
image.save("output.png")

# 可选：移除补丁
remove_patch(pipe)
```

### 示例2：在扩散过程中动态应用

```python
from AgentAttention_block import apply_patch, remove_patch

def custom_sampling_loop(model, prompt, num_inference_steps=50):
    # 初始化
    latents = ...
    
    for i, t in enumerate(timesteps):
        # 在早期步骤应用强合并
        if i == 0:
            remove_patch(model)  # 先移除之前的补丁
            apply_patch(model, ratio=0.4, agent_ratio=0.95, sx=4, sy=4)
        
        # 在中期步骤调整参数
        elif i == num_inference_steps // 2:
            remove_patch(model)
            apply_patch(model, ratio=0.4, agent_ratio=0.7, sx=2, sy=2)
        
        # 执行扩散步骤
        noise_pred = model(latents, t, prompt)
        latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    return latents
```

### 示例3：性能对比测试

```python
import time
from AgentAttention_block import apply_patch, remove_patch

# 测试原始模型
start_time = time.time()
for _ in range(10):
    image = pipe(prompt).images[0]
original_time = (time.time() - start_time) / 10

# 应用补丁
apply_patch(pipe, ratio=0.4, agent_ratio=0.8)

# 测试加速后模型
start_time = time.time()
for _ in range(10):
    image = pipe(prompt).images[0]
accelerated_time = (time.time() - start_time) / 10

print(f"原始模型: {original_time:.2f}s/张")
print(f"加速后模型: {accelerated_time:.2f}s/张")
print(f"加速比: {original_time/accelerated_time:.2f}x")
```

## ⚙️ 参数调优指南

### ratio (Token合并比例)

- **0.2-0.3**：保守设置，质量优先，速度提升较小
- **0.4-0.5**：平衡设置，推荐使用（**默认推荐**）
- **0.6-0.7**：激进设置，速度优先，可能影响质量

### agent_ratio (Agent token比例)

- **0.7-0.8**：标准设置，推荐使用（**默认推荐**）
- **0.9-0.95**：早期步骤使用，更强压缩
- **0.5-0.6**：后期步骤使用，保持细节

### max_downsample (最大下采样层数)

- **1**：仅在无下采样层应用（推荐，质量最好）
- **2**：允许2倍下采样层应用（平衡）
- **4-8**：在所有层应用（速度最快，可能影响质量）

### attn_precision (注意力精度)

- **None**：使用模型默认精度（SD v1.5）
- **"fp32"**：使用FP32精度（**SD v2.1推荐**，避免数值不稳定）

## 🔍 常见问题

### Q1: 模块导入失败怎么办？

**A:** 确保 `agentsd` 目录存在且包含 `merge.py` 和 `utils.py` 文件。如果还是失败，检查 Python 路径设置。

```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
```

### Q2: 应用补丁后模型输出异常？

**A:** 尝试以下方法：
1. 检查参数设置是否合理（ratio不要过大）
2. 对于 SD v2.1，设置 `attn_precision="fp32"`
3. 尝试降低 `max_downsample` 值
4. 确保在应用补丁前模型已正确加载

### Q3: 如何在不同阶段使用不同参数？

**A:** 在扩散循环中使用 `remove_patch()` 和 `apply_patch()` 动态切换参数：

```python
if step == 0:
    remove_patch(model)
    apply_patch(model, ratio=0.4, agent_ratio=0.95)
elif step == 20:
    remove_patch(model)
    apply_patch(model, ratio=0.4, agent_ratio=0.5)
```

### Q4: 内存使用没有明显减少？

**A:** 
1. 增加 `ratio` 值（但不要超过0.6）
2. 增加 `max_downsample` 值
3. 确保 `merge_attn=True`（默认已开启）

### Q5: 生成速度没有明显提升？

**A:**
1. 检查是否正确应用了补丁
2. 增加 `ratio` 和 `agent_ratio` 值
3. 使用更大的 `sx` 和 `sy` 值
4. 确保在GPU上运行

### Q6: 支持哪些模型格式？

**A:** 
- ✅ Stable Diffusion v1.5 (LDM格式)
- ✅ Stable Diffusion v2.0/v2.1 (LDM格式)
- ✅ Diffusers库的Stable Diffusion模型
- ✅ 其他基于Transformer Block的扩散模型

## 📊 性能参考

根据论文和测试，使用推荐参数（ratio=0.4, agent_ratio=0.8）在 Stable Diffusion v1.5 上：

- **速度提升**: 1.3-1.7x 加速
- **内存减少**: 1.5-2.0x 减少
- **质量提升**: FID分数降低 0.7-1.0（更好的质量）

## 🛠️ 高级用法

### 自定义合并策略

```python
# 只在自注意力层合并（推荐）
apply_patch(model, merge_attn=True, merge_crossattn=False, merge_mlp=False)

# 在自注意力和交叉注意力层都合并
apply_patch(model, merge_attn=True, merge_crossattn=True, merge_mlp=False)

# 在所有层合并（不推荐，可能影响质量）
apply_patch(model, merge_attn=True, merge_crossattn=True, merge_mlp=True)
```

### 检查补丁状态

```python
# 检查模型是否已应用补丁
def is_patched(model):
    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            return True
    return False

if is_patched(model):
    print("模型已应用Agent Attention补丁")
else:
    print("模型未应用补丁")
```

## 📖 参考资料

- 论文: [Agent Attention: On the Integration of Softmax and Linear Attention](https://arxiv.org/abs/2312.08874)
- 原始项目: [Agent-Attention](https://github.com/...)
- ToMeSD: [Token Merging for Stable Diffusion](https://github.com/dbolya/tomesd)

## 📄 许可证

请参考原始项目的许可证文件。

## 🤝 贡献

如有问题或建议，欢迎提交Issue或Pull Request。

---

**祝使用愉快！如有任何问题，请参考常见问题部分或查看代码注释。**

