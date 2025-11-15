# 🎯 即插即用模块最新进展 | Papers & Codes



## 📖 介绍 (Introduction)

本仓库系统性地整理了目标检测及相关视觉领域的顶会（CVPR, ICCV, ECCV, NeurIPS等）与顶刊论文，并附注其官方开源代码。

## 🎯 目标 (Goal)

帮助研究人员和开发者快速跟进前沿技术，便捷地复现和比较SOTA方法。

## ✨ 特点 (Features)

- 📂 **论文速递** 收录最新顶会顶刊论文
- 💻 **代码链接** 直接指向官方开源实现
- 🏆 **论文概述** 解读论文核心技术亮点
- 🔄 **持续更新** 紧跟学术前沿定期维护

---

## 📚 论文索引 (Papers Index)

本仓库根据技术创新点分为以下几个大类，你可以点击`[详情]`快速跳转到相应版块。

* [**01. 注意力机制 (Attention)**](#01_Attention)
* [**02. 卷积创新 (Convolution)**](#02_Convolution)
* [**03. 重参数化 (Re_Parameterization)**](#03_Re_Parameterization)
* [**04. Transformer系列 (Transformer)**](#04_Transformer)
* [**05. 特征融合/Neck (Feature_Fusion_Neck)**](#05_Feature_Fusion_Neck)
* [**06. 频域与空域 (Frequency_Spatial)**](#06_Frequency_Spatial)
* [**07. 激活与归一化 (Activation_Normalization)**](#07_Activation_Normalization)

---

### <a id="01_Attention"></a>📂 01_Attention (注意力机制)

本章节汇总了各种新颖的注意力机制，它们通过增强关键特征的表达来提升模型性能。

* **ABC-Attention**
    * **简要说明:** 提出了一种用于红外小目标检测的注意力机制。它基于Transformer架构，通过新颖的双线性相关（Bilinear Correlation）注意力来有效增强微小目标特征并抑制背景噪声。
    * **链接:** [[Paper]](https://arxiv.org/abs/2303.10321) [[Code]](https://github.com/PANPEIWEN/ABC) (CVPR 2024)
* **Agent-Attention**
    * **简要说明:** 提出了一种新颖的“代理注意力”范式。它引入一小组“代理令牌”（Agent Tokens）来负责聚合和广播全局信息，旨在平衡Transformer中Softmax注意力的强大表达能力和线性注意力的计算效率。
    * **链接:** [[Paper]](https://arxiv.org/abs/2312.08874) [[Code]](https://github.com/LeapLabTHU/Agent-Attention) (ECCV 2024)
* **Attention-GhostUNet++**
    * **简要说明:** 一种用于医学图像分割的深度学习模型。它专注于精确分割CT图像中的腹部脂肪组织（皮下和内脏）以及肝脏。
    * **链接:** [[Paper]](https://arxiv.org/abs/2504.11491) [[Code]](https://github.com/MansoorHayat777/Attention-GhostUNetPlusPlus) (Sensors 2024)
* **DarkIR**
    * **简要说明:** 一种用于低光图像恢复的多任务模型，能同时处理噪声、低光和模糊。它没有使用Transformer，而是通过新颖的加法注意力机制（Additive Attention）来增强CNN的性能，以更低的计算成本实现了SOTA效果。
    * **链接:** [[Paper]](https://arxiv.org/abs/2412.13443) [[Code]](https://github.com/cidautai/DarkIR) (CVPR 2025)
* **PFT-SR**
    * **简要说明:** 提出了一种用于单图像超分辨率的“渐进式聚焦Transformer”（Progressive Focused Transformer）。它使用渐进式聚焦注意力（PFA）来链接网络中的注意力图，使模型能专注于最重要的特征，并通过过滤无关特征来降低计算成本。
    * **链接:** [[Paper]](https://arxiv.org/abs/2503.20337) [[Code]](https://github.com/LabShuHangGU/PFT-SR) (CVPR 2025)
* **SvANet**
    * **简要说明:** 一种用于微小医疗对象分割的“尺度变化注意力网络”。它引入了蒙特卡洛注意力（MCAttn）和尺度变化注意力（SvAttn）来处理不同尺度的小目标，能跨多种医疗图像模态（如CT, MRI等）工作。
    * **链接:** [[Paper]](https://arxiv.org/abs/2407.07720) [[Code]](https://github.com/anthonyweidai/SvANet) (WACV 2025)
* **SCSA**
    * **简要说明:** 一种即插即用的“空间与通道协同注意力”（SCSA）模块 。它旨在探索空间和通道注意力之间被忽视的“协同效应” 。它引入了 `SMSA`（共享多语义空间注意力），利用多尺度 1D 卷积来提取不同的空间语义信息，并以此“指导” `PCSA`（渐进式通道自注意力） 。PCSA 进而使用通道自注意力来“缓解”不同语义特征间的“差异” ，从而在分类、检测和分割等多种视觉任务上实现更好的特征融合。
    * **链接:** [[Paper]](https://arxiv.org/abs/2407.05128) [[Code]](https://github.com/HZAI-ZJNU/SCSA) (SCI一区 2025)
* **AMDNet**
    * **简要说明：** 一种用于时间序列预测（TSF）的 **AMD（自适应多尺度分解）**框架。该方法基于 MLP，旨在解决 Transformer 的“过拟合”和 MLP 的“信息瓶颈”问题。其核心是**“分解-交互-合成”**：它首先通过 `MDM` 模块将时间序列**分解**为多个尺度；然后通过 `DDI` 模块高效建模其时序和通道依赖；最后，它创新地使用了一个**混合专家（MoE）**架构——即 `AMS` 模块——来对这些不同尺度的预测进行**自适应加权**，从而实现高效且鲁棒的预测。
    * **链接:** [[Paper]](https://arxiv.org/pdf/2406.03751) [[Code]](https://github.com/TROUBADOUR000/AMD) (AAAI 2025)
    

---

### <a id="02_Convolution"></a>📂 02_Convolution (卷积创新)

本章节汇总了对标准卷积操作的改进，包括动态卷积、大核卷积、可变形卷积等。

* **deformableLKA**
    * **简要说明:** 全称为“Beyond Self-Attention: Deformable Large Kernel Attention”，用于3D医学图像分割。它提出“可变形大核注意力”（D-LKA），使用大的、可变形的卷积核来捕捉上下文信息。这实现了类似自注意力的感受野，但计算开销远低于Transformer。
    * **链接:** [[Paper]](https://arxiv.org/abs/2309.00121) [[Code]](https://github.com/xmindflow/deformableLKA) (WACV 2024)

* **LDConv**
    * **简要说明:** 提出“线性可变形卷积”（Linear Deformable Convolution）。与参数量随核大小呈平方增长的标准卷积不同，LDConv的参数量呈线性增长。它使用不规则和可变形的采样形状来动态适应目标，可作为标准卷积的轻量级替代。
    * **链接:** [[Paper]](https://arxiv.org/abs/2311.11587) [[Code]](https://github.com/CV-ZhangXin/LDConv) (Image Vis. Comput 2023)

**ARConv**

* **简要说明：** 一种用于遥感图像全色锐化的新型卷积模块，名为 **ARConv（自适应矩形卷积）**。它旨在打破传统卷积核“固定方形”和“固定采样点数”的束缚。ARConv 能够**自适应地学习**卷积核所需的高度（$h$）和宽度（$w$），并根据学习到的尺度**动态调整采样点的数量**（$k_h, k_w$）。这种设计使其能够灵活地从像素级别捕捉遥感图像中尺度和形状各异的地物特征。
* **链接:** [[Paper]](https://arxiv.org/abs/2503.00467) [[Code]](https://github.com/WangXueyang-uestc/ARConv) (CVPR 2025)

---

### <a id="03_Re_Parameterization"></a>📂 03_Re_Parameterization (重参数化)

本章节汇总了“重参数化”技术的论文，这些技术在训练时使用复杂结构，在推理时等效融合为简单结构，以实现性能和速度的平衡。
* *(待补充...)*

---

### <a id="04_Transformer"></a>📂 04_Transformer (Transformer系列)

本章节汇总了基于Transformer架构的视觉模型和模块。

* **AST-Transformer**
    * **简要说明:** 全称为“Adapt or Perish: Adaptive Sparse Transformer”，用于图像恢复。它提出“自适应稀疏Transformer”（AST），使用自适应稀疏自注意力（ASSA）块来过滤掉无关区域的噪声干扰，并使用特征精炼前馈网络（FRFN）来减少特征冗余。
    * **链接:** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.html) [[Code]](https://github.com/joshyZhou/AST) (CVPR 2024)

* **SwiftFormer**
    * **简要说明:** 一种为移动端设计的实时、高效Transformer模型。它提出“高效加法注意力”，用线性的逐元素乘法取代了昂贵的矩阵乘法，使得注意力模块可以用在网络的所有阶段，实现了极佳的速度-精度权衡。
    * **链接:** [[Paper]](https://arxiv.org/abs/2303.15446) [[Code]](https://github.com/Amshaker/SwiftFormer) (ICCV 2023)

* **WPFormer**
    * **简要说明:** 一种用于像素级表面缺陷检测的查询式Transformer。它提出了“小波和原型增强查询”（Wavelet and Prototype Augmented Query），利用双域（频域和时域）解码器和小波增强通道注意力（WCA）来使模型更关注高频细节，有效检测工业表面的微小缺陷。
    * **链接:** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html) [[Code]](https://github.com/iefengyan/WPFormer) (CVPR 2025)

* **DAT-Transformer**
    * **简要说明:** 提出“可变形注意力Transformer”（Deformable-Attention-based Transformer）。其核心是可变形的自注意力模块，其中“键”和“值”对的位置是根据数据动态学习的，这使得模型能够自适应地聚焦于相关的图像区域，捕获更多信息特征。
    * **链接:** [[Paper]](https://arxiv.org/abs/2308.03364) [[Code]](https://github.com/zhengchen1999/DAT) (ICCV 2023)

* **TBSN**
    * **简要说明:**这篇论文提出了一种名为 **TBSN** 的新型自监督去噪网络 。它通过巧妙地**重新设计 Transformer 的注意力机制**（G-CSA 和 M-WSA），使其能够**在满足“盲点”约束**（BSN）的前提下工作 。这样，模型既能利用 Transformer 强大的全局感知能力，又不会“偷看”到原始噪声，从而在自监督去噪任务上达到了 SOTA 性能。
    * **链接:** [[Paper]](https://arxiv.org/abs/2404.07846) [[Code]](https://github.com/nagejacob/TBSN) (AAAI 2025)

---

### <a id="05_Feature_Fusion_Neck"></a>📂 05_Feature_Fusion_Neck (特征融合/Neck系列)

本章节汇总了FPN及其他用于融合多尺度特征的Neck结构创新。
* *(待补充...)*

---

### <a id="06_Frequency_Spatial"></a>📂 06_Frequency_Spatial (频域与空域转换系列)

本章节汇总了结合傅里叶变换等频域处理与空域信息的模型。

* **ASCNet**
    * **简要说明:** 提出“注意力扩张空间光谱卷积网络”，用于高光谱图像（HSI）分类。该模型包含一个用于学习空间特征的注意力CNN分支和一个用于学习光谱相关性的注意力RNN分支，最后将两者联合学习。
    * **链接:** [[Paper]](https://arxiv.org/abs/2401.15578) [[Code]](https://github.com/xdFai/ASCNet) (CVPR 2025)

---

### <a id="07_Activation_Normalization"></a>📂 07_Activation_Normalization (激活函数与归一化系列)

本章节汇总了对ReLU、BatchNorm等模块的改进。
* MONA
  * **简要说明:** 一种名为 SCSA (空间与通道协同注意力) 的即插即用注意力模块。它旨在探索空间和通道注意力之间被忽视的“协同效应” ，即现有方法未能解决的“多语义差异”问题。它引入了 `SMSA`（共享多语义空间注意力） ，利用多尺度 1D 卷积来提取不同的空间语义信息 ，并以此作为空间先验来“指导” `PCSA`（渐进式通道自注意力） 。PCSA 进而使用通道自注意力来“缓解”不同语义特征间的“差异” ，从而在分类、检测和分割等多种视觉任务上实现更优的特征融合 。。
  * **链接:** [[Paper]](https://arxiv.org/abs/2408.08345) [[Code]](https://github.com/LeiyiHU/mona) (CVPR 2025)

## 📜 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源。
