## 论文精度：AMDNet

### 1. 核心思想

* 本文提出了一种名为 **AMD（自适应多尺度分解）**的 MLP-based 框架，专用于时间序列预测（TSF）。
* 其核心思想是，现实世界的时间序列具有复杂的**“多尺度纠缠”（multi-scale entanglement）**特性，而现有的 Transformer 方法（计算昂贵且易过拟合）和 MLP 方法（过于简单）都无法有效建模这一点。
* AMD 框架通过 **MDM 模块**将时间序列**分解**为多个不同尺度的子序列，通过 **DDI 模块**高效建模这些子序列的时序和通道依赖，最后通过 **AMS 模块**（一个 MoE 混合专家模型）对这些不同尺度的预测进行**自适应加权**。
* 这种“分解-交互-自适应合成”的策略，使得 AMD 作为一个 MLP-based 架构，在保持高效率（线性复杂度）的同时，首次在性能上**全面超越**了 SOTA Transformer 模型（如 PatchTST, iTransformer）。

### 2. 背景与动机

* **[文本角度总结]**
    时间序列预测（TSF）领域目前由 Transformer-based 和 MLP-based 两类方法主导，但两者都存在显著缺陷：
    
    1.  **Transformer-based 方法（如 PatchTST）**：
        * **优点**：擅长捕捉长程依赖。
        * **缺点（效率瓶颈）**：自注意力机制具有 $O(N^2)$ 的**平方计算复杂度**，导致训练效率低、内存消耗大。
        * **缺点（语义鸿沟）**：自注意力机制倾向于**过度关注“突变点”**（Mutation Points），而忽视了平滑的、连续的**时序动态**（temporal dynamics），导致**过拟合**（如图 1 所示）。
    2.  **MLP-based 方法（如 DLinear）**：
        * **优点**：计算效率极高（线性复杂度），擅长建模时序动态。
        * **缺点（语义鸿沟）**：由于其简单的线性映射，存在“**信息瓶颈**”（information bottleneck），难以捕捉和区分现实世界中复杂且**纠缠在一起的多尺度时间模式**（例如，每小时的天气波动 vs. 每月的气候趋势）。
    
    **本文的动机**：设计一个新框架，既能拥有 MLP 的**高效率**和**时序建模能力**，又能克服其“信息瓶颈”，使其能像 Transformer 一样捕捉和建模**复杂的多尺度模式**。
    
* **动机图解分析（Figure 1 & 4）：**
    * **图表 A (Figure 1)：揭示“多尺度纠缠”与“过拟合”问题**
        * **“看图说话”：** 这张图是本文的核心动机。左侧的“Historical Input”被（概念上）分解为三种不同尺度的序列：“Coarse-Grained”（粗粒度/趋势）、“Fine-Grained”（细粒度/噪声）和中尺度。
        * **分析（语义鸿沟）：** 现实世界（如右侧 `Predict Series`）的未来变化是由**所有这些尺度的纠缠**共同决定的。而现有的 MLP 太简单，无法有效分离这些尺度。
        * **分析（效率瓶颈/过拟合）：** 图的左下角展示了 Transformer 的问题。`High Attention Score`（高注意力分数）**过度聚焦于“Mutate Points”（突变点/异常值）**。这导致模型学到的是“噪声”而非“模式”，从而在预测（Predict Series）时产生过拟合，无法捕捉到真实的周期性。
        * **结论：** Figure 1 提出了两个核心挑战：1) 必须对信号进行**多尺度分解**；2) 必须**自适应地聚合**这些尺度，而不是像 Transformer 那样过拟合于突变点。

    * **图表 B (Figure 4)：揭示“通道依赖”的“效率瓶颈”**
        * **“看图说话”：** 这张图对比了引入“跨通道依赖”（Cross-channel dependencies）前后的特征分布热力图。
        * **分析：** “Before”（左图）是仅考虑时序依赖的特征分布。“After”（右图）是在引入跨通道依赖（即让不同变量相互影响）后的分布。可以清晰地看到，“After”的特征分布被**过度平滑**了，导致其**偏离了原始分布**。
        * **结论：** 这揭示了一个“效率瓶颈”或“语义鸿沟”：在多元时间序列中，**天真地混合所有通道（变量）的信息是有害的**，因为它会引入不相关变量的“噪声”，污染目标变量的特征。这直接催生了本文 `DDI` 模块的设计——它必须有一个**控制机制（$\beta$ 缩放系数）**来“缓解”这种有害的通道交互。

### 3. 主要贡献点

1.  **提出 AMD 框架：** 提出了一个新颖的、完全基于 MLP 的自适应多尺度分解框架（AMD）。它**摒弃了 Transformer 的自注意力机制**，通过“分解-交互-合成”三阶段解决了 MLP 无法处理多尺度模式的“信息瓶颈”问题。
2.  **发明 MDM 模块（多尺度分解混合）：**
    * 这是**分解**阶段。`MDM` 模块使用**平均下采样**（AvgPooling）将单条时间序列分解为 $h$ 个不同尺度（$\tau_1, \dots, \tau_h$）的子序列（即时间模式）。
    * 接着，它通过一个**从粗到细（coarse-to-fine）**的**残差 MLP** 路径（$\xi_i = \tau_i + MLP(\xi_{i+1})$）来**混合**这些尺度，使得细粒度特征（$\tau_1$）能够感知到粗粒度（$\xi_2$）的上下文。
3.  **发明 DDI 模块（双重依赖交互）：**
    * 这是**交互**阶段。`DDI` 是一个高效的 MLP 块，用于处理 `MDM` 混合后的特征。
    * 它通过两个并行的 MLP（一个作用于时间步，一个作用于通道）来**同时建模“时序依赖”（temporal dependencies）和“通道依赖”（channel dependencies）**。
    * 关键是，它引入了一个**缩放系数 $\beta$** 来**控制通道交互的强度**，防止不相关的变量相互干扰（解决了 Figure 4 所示的问题）。
4.  **发明 AMS 模块（自适应多预测器合成）：**
    * 这是**合成**阶段，也是本文**最核心的创新**。它本质上是一个**混合专家（MoE）**架构。
    * `AMS` 包含两个组件：一个 **`TP-Selector`（门控网络）**和 $m$ 个并行的 **`Predictor`（专家网络）**。
    * `TP-Selector` 负责分析 `MDM` 提供的多尺度信息，**动态生成“选择器权重” $S$**（即决定每个尺度/模式对未来预测的“重要性”）。
    * $m$ 个 `Predictor` 则分别对 `DDI` 处理后的特征进行独立预测。
    * 最终输出是所有 $m$ 个预测的**加权和**（$\hat{Y} = \sum S_j \cdot Predictor_j(v)$）。这种 MoE 机制使得 AMD 能**自适应地聚焦于“主导的时间模式”**，而忽略噪声和突变点（解决了 Figure 1 所示的 Transformer 过拟合问题）。

### 4. 方法细节

* **整体网络架构（Figure 2）：**
    
    ![结构图2](https://gitee.com/ChadHui/typora-image/raw/master/cv-image/20251115194129.jpg)
    
    * **模型名称：** AMD (Adaptive Multi-Scale Decomposition)
    * **数据流：** 这是一个**三阶段的串行（Sequential）**架构，完全由 MLP 及其变体构成。
    * **输入：** $X$（$C \times L$），首先经过 `RevIN`（可逆实例归一化）处理。
    * **阶段 1：`Multi-Scale Decomposable Mixing` (MDM 块 - 分解)：**
        * 输入 $X$（逐通道处理，得到 $u$）进入该模块。
        * **下采样：** 输入 $\tau_1$ (原始序列) 被 `Down Sampling`（AvgPooling） 递归 $h$ 次，产生 $h$ 个不同尺度的序列 $\tau_1, \tau_2, \dots, \tau_h$。
        * **混合：** 从最粗粒度的 $\tau_h$ 开始，通过 `MLP` 向上（从粗到细）进行残差混合。$\xi_h = \tau_h$，然后 $\xi_i = \tau_i + MLP(\xi_{i+1})$。
        * **输出：** 最终混合了所有尺度信息的特征 $\xi_1$（记为 $u$）被输出。
    * **阶段 2：`Dual Dependency Interaction` (DDI 块 - 交互)：**
        * **堆叠：** 来自 MDM 的 $C$ 个 $u$（$1 \times L$）被**堆叠**（Stack Channel-Wise）成一个 $U$（$C \times L$）矩阵。
        * **打补丁 (Patch)：** $U$ 被切分为 $N$ 个 Patch。
        * **混合：** 执行 MLP-Mixer 风格的双重依赖交互（时序 MLP + 通道 MLP + $\beta$ 缩放）。
        * **输出：** 得到 $V$（$C \times L$），并**拆分**（Split Channel-Wise）为 $C$ 个 $v$（$1 \times L$）输出。
    * **阶段 3：`Adaptive Multi-predictor Synthesis` (AMS 块 - 合成)：**
        * 这是一个 MoE 模块，**同时接收**来自 MDM 的 $u$ 和来自 DDI 的 $v$。
        * **门控路径（TP-Selector）：** $u$（来自 MDM）进入 `TP-Selector`。`Decomp. & Score` 模块（包含 `TopK` 和 `Softmax`）生成 `Selector Weights` $S$（$m \times T$ 矩阵，$m$ 为专家数，$T$ 为预测长度）。
        * **专家路径（TP-Projection）：** $v$（来自 DDI）进入 `TP-Projection`。它被**并行**送入 $m$ 个 `Predictor` 块（每个都是 MLP）。
        * **聚合：** $m$ 个预测结果根据 `Selector Weights` $S$ 进行**加权求和（Weighted Sum）**，得到最终的 $1 \times T$ 预测 $\tilde{y}$。
    * **输出：** 所有通道的预测 $\hat{Y}$ 经过 `RevIN`（反归一化）得到最终结果。
    * **损失函数：** $\mathcal{L}_{total} = \mathcal{L}_{pred} + \lambda_1 \mathcal{L}_{selector} + \lambda_2 ||\Theta||_2$。$\mathcal{L}_{pred}$ 是预测的 MSE 损失，$\mathcal{L}_{selector}$ 是一个 MoE 负载均衡损失，用于防止门控网络“过拟合”于少数几个专家。
    
* **核心创新模块详解：**

    * **对于 模块 A：MDM (Multi-Scale Decomposable Mixing)**
        * **理念：** 将复杂的时序信号分解为多个不同尺度的简单子模式，然后以“从粗到细”的方式将它们重新组合，使高频细节（细粒度）感知到低频趋势（粗粒度）。
        * **数据流：**
            1.  **分解 (Decomposition)：** $\tau_1 = X_{channel}$, $\tau_i = AvgPooling(\tau_{i-1})$。这一步（`Down Sampling`）创建了一个特征金字塔，捕捉了从精细（$\tau_1$）到粗糙（$\tau_h$）的多种时间模式。
            2.  **混合 (Mixing)：** $\xi_h = \tau_h$。$\xi_{i} = \tau_{i} + MLP(\xi_{i+1})$。
        * **设计目的：** 这是对传统分解（如趋势-季节分解）的巨大改进。它不是简单地相加，而是通过一个**残差 MLP** 来学习**跨尺度交互**。这使得模型能够理解“月度趋势（$\xi_{i+1}$）如何非线性地影响日度波动（$\tau_i$）”，从而生成一个对所有尺度都“知情”的特征 $u$。

    * **对于 模块 B：DDI (Dual Dependency Interaction)**
        * **理念：** 高效地（用 MLP）同时建模时序（Temporal）和通道（Channel）依赖，同时**防止通道间噪声干扰**。
        * **数据流：**
            1.  输入 $U$ ($C \times L$) $\rightarrow$ Patching $\rightarrow$ $\hat{U}$ ($C \times N \times P$)。
            2.  **时序混合 (Eq 5)：** $Z = \hat{U} + MLP(\hat{V}_{prev})$。一个 MLP **在 $P$ 维度（时间步）上**操作，捕捉时间依赖性。
            3.  **通道混合 (Eq 6)：** $\hat{V} = Z + \beta \cdot MLP(Z^T)^T$。另一个 MLP **在 $C$ 维度（通道）上**操作（通过转置 $T$ 实现），捕捉通道依赖性。
            4.  **关键创新 ($\beta$)：** $\beta$ 是一个**缩放系数**（scaling rate）。它控制了通道混合（$MLP(Z^T)^T$）对最终特征 $\hat{V}$ 的**贡献度**。
        * **设计目的：** $\beta$ 的存在是为了解决 **Figure 4** 所示的“分布偏移”问题。如果 $\beta$ 很大，模型会过度依赖通道相关性（可能引入噪声）；如果 $\beta$ 很小，模型会退化为“通道独立”（CI）模式，更关注时序。这使得 DDI 模块可以**自适应地平衡**“时序”和“通道”信息。

    * **对于 模块 C：AMS (Adaptive Multi-predictor Synthesis)**
        * **理念：** 这是一个 MoE（混合专家）模块，用于**自适应地**聚合来自不同尺度（由 MDM 提取）的预测。
        * **数据流：**
            1.  **门控（Gating）**：`TP-Selector` 接收 MDM 的输出 $u$（$1 \times L$）。它通过一个 `Decomp. & Score` 模块（包含 MLP 和 TopK）来分析 $u$ 中蕴含的**多尺度模式**。
            2.  `Selector Weights` $S$（$m \times T$）被生成。$S[j, t]$ 代表第 $j$ 个专家（Predictor）对于预测未来第 $t$ 个时间步的“可信度”或“权重”。
            3.  **专家（Experts）**：`TP-Projection` 接收 DDI 的输出 $v$（$1 \times L$）。$v$ 被**并行**送入 $m$ 个独立的 `Predictor` MLP 中。每个 `Predictor_j` 都专精于一种特定的时间模式，并输出一个完整的 $1 \times T$ 预测。
            4.  **合成（Synthesis）**：最终预测 $\tilde{y}$（$1 \times T$）是这 $m$ 个专家预测的**加权平均**：$\tilde{y} = \sum_{j=0}^{m} S_j \cdot Predictor_j(v)$。
        * **设计目的：** `AMS` 解决了 **Figure 1** 所示的“过拟合突变点”问题。Transformer 可能会被某个突变点“欺骗”，而 `AMS` 则通过 `TP-Selector` 来“投票”。`Selector` 会识别出“突变点”只是一种细粒度模式（例如 `Predictor 1`），而“全局趋势”是另一种粗粒度模式（例如 `Predictor 2`）。通过**自适应加权 $S$**，`AMS` 能够更鲁棒地组合这些模式，从而做出更平滑、更准确的预测。

* **理念与机制总结：**
    * AMD 框架在理念上是对 MLP-based TSF 方法的一次重大升级。
    * **DLinear/RLinear** 证明了“单尺度”的 MLP 已经很强。
    * **TimeMixer** 证明了“多尺度分解 + 简单平均”的 MLP 更强。
    * **AMD（本文）** 则证明了“**多尺度分解（MDM） + 自适应加权（AMS/MoE）**”的 MLP 才是最强的。
    * AMD 通过 `MDM` 将复杂问题**分解**为 $h$ 个尺度，然后通过 `AMS`（一个 MoE）**自适应地合成** $m$ 个专家的答案。`DDI` 则在此过程中充当了一个高效的特征交互（时序+通道）模块。
    * 这种“分解-征服-自适应合成”的策略，使得 AMD 作为一个 MLP 家族成员，成功解决了 MLP 的“信息瓶颈”和 Transformer 的“过拟合”问题。

* **图解总结：**
    * **Figure 1** 提出了**问题**：时间序列具有“多尺度纠缠”特性，而 Transformer 会“过拟合突变点”。
    * **Figure 4** 提出了**问题**：盲目的“跨通道”依赖会引入噪声，导致“特征分布偏移”。
    * **Figure 2（左，MDM）** 提供了**解决方案 1**：通过**多尺度分解**（AvgPooling）和**从粗到细的 MLP 混合**，显式地建模“多尺度纠缠”。
    * **Figure 2（中，DDI）** 提供了**解决方案 2**：通过引入**缩放系数 $\beta$**，来**控制**时序混合和通道混合的平衡，解决了“通道噪声”问题。
    * **Figure 2（右，AMS）** 提供了**解决方案 3**：通过 **MoE** 架构（`TP-Selector` + `Predictors`），对 $m$ 个专家的预测进行**自适应加权**，而不是简单平均。这使得模型能聚焦于“主导模式”，避免了对“突变点”的过拟合。

### 5. 即插即用模块的作用

* 本文的 `MDM` 和 `AMS` 模块被明确设计并验证为**即插即用（Plug-and-play）**的组件。

* **作用：** 它们可以作为一个**“性能增强包”**，被集成到**其他现有的 TSF（尤其是 MLP-based）模型**中。

* **适用场景：**
    1.  **增强现有的 MLP-based 模型（如 DLinear, MTS-Mixers）：**
        * **应用：** 如 Table 4 所示，作者将 `DLinear` 和 `MTS-Mixers` 作为基线，并在其架构中**插入**了 `MDM` 和 `AMS` 模块。
        * **优势：** 实验证明，`DLinear + MDM & AMS` 和 `MTS-Mixers + MDM & AMS` 的性能（MSE/MAE）相比原始模型均有显著提升。
        * **结论：** 这表明 `MDM` 提供了原始模型所缺乏的**多尺度分解能力**，而 `AMS` 提供了更强大的**自适应聚合能力**。
    2.  **替换 Transformer 中的注意力机制：**
        * **应用：** 理论上，可以将 Transformer 骨干网络（如 PatchTST）中的“自注意力”块替换为 `MDM + DDI + AMS` 的组合。
        * **优势：** 这将把一个 $O(N^2)$ 复杂度的模型**转换**为一个 $O(N)$ 线性复杂度的模型，同时（如实验所示）可能带来**性能提升**，因为它用 MoE 的自适应聚合替代了自注意力的过拟合倾向。