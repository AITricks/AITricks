## 即插即用 PFA 模块说明（basicsr/archs/pfa_module.py）

本模块将论文图中的“PFA 注意力 + 稀疏计算（SMM）”从整网 `PFT` 中抽出，做成可单独复用的注意力单元，以便在其他骨干网络中即插即用。

### 1. 主要类与功能
- **ProgressiveFocusedAttention**：窗口化多头自注意力（W-MSA/SW-MSA），带“逐层聚焦（PFA）”。
  - 支持按层设定 `top-k` 稀疏化，复用前一层的注意力值/索引实现 progressive focusing。
  - 若已编译 CUDA 扩展 `ops_smm`，则走稀疏矩阵乘（SMM）加速；否则自动回退到纯 PyTorch 实现（功能等价，速度较慢）。

### 2. 环境准备
```bash
pip install -r requirements.txt

# 可选：编译 CUDA 稀疏算子（更快；未编译则自动回退CPU路径）
cd ops_smm
python setup.py install    # 或 pip install -e .
```

### 3. 张量形状与参数
- 设窗口尺寸为 `W`，则 `N = W*W`；通道为 `C`，头数为 `H`。
- 前向调用签名：
```python
out, pfa_values, pfa_indices = attn(qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0)
```
- 参数说明：
  - `qkvp`: `(B*nw, N, 4C)`，按 `[Q, K, V, V_lepe]` 拼接后的张量。
  - `pfa_values/pfa_indices`: `list`，长度为 2（未移位/移位），用于跨层传递 PFA 值与索引；初次可传 `[None, None]`。
  - `rpi`: `(N, N)` 相对位置索引（可复用 `PFT.calculate_rpi_sa()` 生成）。
  - `mask`: 仅在 `shift=1`（SW-MSA）使用；普通 W-MSA 传 `None`。
  - `shift`: `0/1`，指示未移位/移位分支。
  - 返回：`out: (B*nw, N, C)` 与更新后的 `pfa_values/pfa_indices`。

### 4. 前向流程要点
- 计算 `QK^T`：
  - 无稀疏索引时，走密集路径并加相对位置偏置；
  - 有稀疏索引时，仅在保留的位置上计算（`SMM_QmK`/fallback）。
- Softmax 后，用上一层的 `pfa_values` 做 Hadamard 重加权并归一化。
- 按层执行 `top-k` 稀疏化，更新 `pfa_indices` 以供下一层复用（实现“Progressive Focused”）。
- 计算 `A @ V`：
  - 无稀疏索引时走密集 matmul；有索引时走 `SMM_AmV`/fallback；最后与 `V_lepe` 相加。

### 5. 快速自测
模块内置 `main`，随机张量前向一次即可验证：
```bash
conda activate torchv5
python basicsr/archs/pfa_module.py
```
输出示例：
```
Output shape: (2, 64, 64)
PFA map shape: (2, 8, 64, 64)
Indices present: False
Using CUDA SMM: False
```

### 6. 在你的网络中集成
```python
from basicsr.archs.pfa_module import ProgressiveFocusedAttention

attn = ProgressiveFocusedAttention(dim=96, layer_id=0, window_size=8,
                                   num_heads=6, num_topk=[64]*24)

# 准备好 qkvp / pfa_values / pfa_indices / rpi / mask ...
out, pfa_values, pfa_indices = attn(qkvp, pfa_values, pfa_indices, rpi, mask=None, shift=0)
```

小贴士：若你在现有 `pft_arch.py` 的 `WindowAttention` 位置替换为本模块，也能复用相同的 `rpi/mask` 与跨层 PFA 缓存逻辑。

### 7. 常见问题
- 没有安装 `smm_cuda` 可以用吗？
  - 可以。模块自动回退到纯 PyTorch 的 gather/scatter 实现，功能一致但速度会慢。
- `num_topk` 怎么设置？
  - 传入一个按“全局层号”排布的列表。例如 24 层就传 24 个值；不需要稀疏化时将对应位置设为 `N`（等于窗口 token 数）。
- 形状错位/维度不匹配？
  - 对齐 `qkvp: (B*nw, N, 4C)`、`rpi: (N, N)`、`pfa_*` 列表长度为 2，并确保 `window_size` 与你的分块逻辑一致。


