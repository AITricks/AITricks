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

* <u>**ABC-Attention**</u>
    * **简要说明:** 提出了一种用于红外小目标检测的注意力机制。它基于Transformer架构，通过新颖的双线性相关（Bilinear Correlation）注意力来有效增强微小目标特征并抑制背景噪声。
    * **链接:** [[Paper]](https://arxiv.org/abs/2303.10321) [[Code]](https://github.com/PANPEIWEN/ABC) (CVPR 2024)
* **<u>Agent-Attention</u>**
    * **简要说明:** 提出了一种新颖的“代理注意力”范式。它引入一小组“代理令牌”（Agent Tokens）来负责聚合和广播全局信息，旨在平衡Transformer中Softmax注意力的强大表达能力和线性注意力的计算效率。
    * **链接:** [[Paper]](https://arxiv.org/abs/2312.08874) [[Code]](https://github.com/LeapLabTHU/Agent-Attention) (ECCV 2024)
* <u>**Attention-GhostUNet++**</u>
    * **简要说明:** 一种用于医学图像分割的深度学习模型。它专注于精确分割CT图像中的腹部脂肪组织（皮下和内脏）以及肝脏。
    * **链接:** [[Paper]](https://arxiv.org/abs/2504.11491) [[Code]](https://github.com/MansoorHayat777/Attention-GhostUNetPlusPlus) (Sensors 2024)
* <u>**DarkIR**</u>
    * **简要说明:** 一种用于低光图像恢复的多任务模型，能同时处理噪声、低光和模糊。它没有使用Transformer，而是通过新颖的加法注意力机制（Additive Attention）来增强CNN的性能，以更低的计算成本实现了SOTA效果。
    * **链接:** [[Paper]](https://arxiv.org/abs/2412.13443) [[Code]](https://github.com/cidautai/DarkIR) (CVPR 2025)
* <u>**PFT-SR**</u>
    * **简要说明:** 提出了一种用于单图像超分辨率的“渐进式聚焦Transformer”（Progressive Focused Transformer）。它使用渐进式聚焦注意力（PFA）来链接网络中的注意力图，使模型能专注于最重要的特征，并通过过滤无关特征来降低计算成本。
    * **链接:** [[Paper]](https://arxiv.org/abs/2503.20337) [[Code]](https://github.com/LabShuHangGU/PFT-SR) (CVPR 2025)
* <u>**SvANet**</u>
    * **简要说明:** 一种用于微小医疗对象分割的“尺度变化注意力网络”。它引入了蒙特卡洛注意力（MCAttn）和尺度变化注意力（SvAttn）来处理不同尺度的小目标，能跨多种医疗图像模态（如CT, MRI等）工作。
    * **链接:** [[Paper]](https://arxiv.org/abs/2407.07720) [[Code]](https://github.com/anthonyweidai/SvANet) (WACV 2025)
* <u>**SCSA**</u>
    * **简要说明:** 一种即插即用的“空间与通道协同注意力”（SCSA）模块 。它旨在探索空间和通道注意力之间被忽视的“协同效应” 。它引入了 `SMSA`（共享多语义空间注意力），利用多尺度 1D 卷积来提取不同的空间语义信息，并以此“指导” `PCSA`（渐进式通道自注意力） 。PCSA 进而使用通道自注意力来“缓解”不同语义特征间的“差异” ，从而在分类、检测和分割等多种视觉任务上实现更好的特征融合。
    * **链接:** [[Paper]](https://arxiv.org/abs/2407.05128) [[Code]](https://github.com/HZAI-ZJNU/SCSA) (SCI一区 2025)
* <u>**AMDNet**</u>
    * **简要说明：** 一种用于时间序列预测（TSF）的 **AMD（自适应多尺度分解）**框架。该方法基于 MLP，旨在解决 Transformer 的“过拟合”和 MLP 的“信息瓶颈”问题。其核心是**“分解-交互-合成”**：它首先通过 `MDM` 模块将时间序列**分解**为多个尺度；然后通过 `DDI` 模块高效建模其时序和通道依赖；最后，它创新地使用了一个**混合专家（MoE）**架构——即 `AMS` 模块——来对这些不同尺度的预测进行**自适应加权**，从而实现高效且鲁棒的预测。
    * **链接:** [[Paper]](https://arxiv.org/pdf/2406.03751) [[Code]](https://github.com/TROUBADOUR000/AMD) (AAAI 2025)

* <u>**EfficientViM**</u>
    * **简要说明：** Efficient ViM 针对标准 Mamba 模型在视觉任务中线性投影计算成本过高的问题，提出了一种 基于隐状态混合器的状态空间对偶 (HSM-SSD) 模块。该模块巧妙地将昂贵的通道混合与门控操作转移到压缩后的隐状态空间进行，大幅降低了计算复杂度，同时保留了全局建模能力。配合单头设计消除内存瓶颈，以及 多阶段隐状态融合 (MSF) 策略增强特征表征，Efficient ViM 在 ImageNet 上实现了超越 MobileNetV3 和 SHVIT 的速度-精度权衡，是构建高效轻量级视觉主干网络的理想选择。
    * **链接:** [[Paper]](https://arxiv.org/pdf/2411.15241) [[Code]](https://github.com/mlvlab/EfficientViM) (AAAI 2025)

* <u>**MobileMamba**</u>
    * **简要说明：** MobileMamba 提出了一种高效的轻量级视觉网络，旨在解决 CNN 感受野受限和 ViT 计算复杂度高的问题。其核心即插即用模块 MRFFI (多感受野特征交互) 创新性地结合了 WTE-Mamba 和 MK-DeConv。WTE-Mamba 利用小波变换增强 Mamba 对高频细节的捕捉能力并进行全局建模；MK-DeConv 通过多核深度卷积并行提取多尺度局部特征。这种“分而治之”的策略在保持线性计算复杂度的同时，实现了全局与局部信息的有效融合，显著提升了轻量级模型在图像分类及下游任务中的性能与推理速度。
    * **链接:** [[Paper]](https://arxiv.org/abs/2411.15941) [[Code]](https://github.com/lewandofskee/MobileMamba) (CVPR 2025)

* <u>**HVI-CIDNet**</u>
  * **简要说明：** HVI-CIDNet 针对低光照图像增强中颜色失真和噪声放大的问题，提出了一种创新的解决方案。它首先通过 HVI 颜色空间变换，在输入端解耦亮度和色度，并利用极坐标化和强度塌缩机制从物理/数学层面消除红黑噪声伪影。随后，CIDNet 利用双分支架构和 LCA (轻量级交叉注意力) 模块，分别处理并交互融合亮度和色度特征。这种“空间变换+双流解耦”的策略，不仅大幅降低了计算成本，还在多个数据集上实现了 SOTA 的增强效果。
    * **链接:** [[Paper]](https://arxiv.org/abs/2502.20272) [[Code]](https://github.com/Fediory/HVI-CIDNet) (CVPR 2025)
  
* <u>**PG-DRFNet**</u>
    * **简要说明：**  PG-DRFNet 针对遥感图像（涵盖光学和 SAR）中小目标检测面临的特征淹没和定位困难问题，提出了一种位置引导的动态感受野网络 。它首先通过 DRF (动态感受野) 模块，在不同特征层之间建立小目标的“位置引导关系” (Positional Guidance Relationship)，利用额外的监督信息重加权特征，有效防止小目标在深层特征中消失。随后，该网络利用由 PGH (位置引导头) 和 DPH (动态感知头) 构成的组合检测头，配合基于特征构建的动态感知算法，在推理阶段动态优化感知区域和特征层级 3333。这种“位置引导+动态感知”的策略，不仅实现了对双模态数据的鲁棒兼容，还在 DOTA-v2.0、VEDAI、SSDD 等四个基准数据集上实现了 SOTA 的检测精度与推理速度的最佳权衡 。
    * **链接:** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10909281) [[Code]](https://github.com/BJUT-AIVBD/PG-DRFNet) (TCSVT 2025)

* <u>**MFF-Net**</u>
    * **简要说明：**   MFF-Net 项目中CrossAttention 负责多尺度或多模态特征的跨注意力融合 ；DirectionalConvUnit 通过多方向卷积操作强化特征的结构信息 ；SWSAM 则结合通道洗牌与分组机制实现轻量级的局部注意力增强。这些模块仅依赖 PyTorch 基础库，可直接嵌入各类网络结构中进行复用 。
    * **链接:** [[Paper]](https://ieeexplore.ieee.org/abstract/document/10892273) [[Code]](https://github.com/HITFuxiwen/MFF-Net) (TCSVT 2025)
    
* <u>**CTO-Net**</u>
    * **简要说明：**  该论文提出了一种名为 **SCSA (Spatial and Channel Synergistic Attention)** 的新型即插即用注意力模块，旨在解决现有混合注意力机制在利用多语义空间信息及处理语义差异方面的不足。SCSA 采用串行结构，首先通过 **SMSA (Shareable Multi-Semantic Spatial Attention)** 模块利用多尺度 1D 卷积提取不同语义的空间特征作为先验；随后通过 **PCSA (Progressive Channel-wise Self-Attention)** 模块利用这些空间先验引导通道自注意力机制，从而有效缓解语义差异并实现特征的深度融合。实验结果表明，SCSA 在图像分类、目标检测和语义分割等多个视觉任务中均取得了优于当前 SOTA 方法（如 CBAM、ECA）的性能表现。
    * **链接:** [[Paper]](https://arxiv.org/pdf/2505.04652v1) [[Code]](https://github.com/xiaofang007/CTO) (MedIA 2025)


---

### <a id="02_Convolution"></a>📂 02_Convolution (卷积创新)

本章节汇总了对标准卷积操作的改进，包括动态卷积、大核卷积、可变形卷积等。

* <u>**deformableLKA**</u>
    * **简要说明:** 全称为“Beyond Self-Attention: Deformable Large Kernel Attention”，用于3D医学图像分割。它提出“可变形大核注意力”（D-LKA），使用大的、可变形的卷积核来捕捉上下文信息。这实现了类似自注意力的感受野，但计算开销远低于Transformer。
    * **链接:** [[Paper]](https://arxiv.org/abs/2309.00121) [[Code]](https://github.com/xmindflow/deformableLKA) (WACV 2024)

* **LDConv**
    * **简要说明:** 提出“线性可变形卷积”（Linear Deformable Convolution）。与参数量随核大小呈平方增长的标准卷积不同，LDConv的参数量呈线性增长。它使用不规则和可变形的采样形状来动态适应目标，可作为标准卷积的轻量级替代。
    * **链接:** [[Paper]](https://arxiv.org/abs/2311.11587) [[Code]](https://github.com/CV-ZhangXin/LDConv) (Image Vis. Comput 2023)

* <u>**ARConv**</u>
    * **简要说明：** 一种用于遥感图像全色锐化的新型卷积模块，名为 **ARConv（自适应矩形卷积）**。它旨在打破传统卷积核“固定方形”和“固定采样点数”的束缚。ARConv 能够**自适应地学习**卷积核所需的高度（$h$）和宽度（$w$），并根据学习到的尺度**动态调整采样点的数量   （$k_h, k_w$）。这种设计使其能够灵活地从像素级别捕捉遥感图像中尺度和形状各异的地物特征。
    * **链接:** [[Paper]](https://arxiv.org/abs/2503.00467) [[Code]](https://github.com/WangXueyang-uestc/ARConv) (CVPR 2025)
    
* <u>**SCConv**</u>
    * **简要说明：** SCConv（空间和通道重建卷积）是一种高效的、即插即用的卷积模块，旨在替代标准卷积层 。该模块包含两个核心单元：SRU（空间重建单元）用于抑制特征的空间冗余 ，CRU（通道重建单元）用于削减通道间的信息冗余。通过这种双重冗余消除，SCConv 能够在显著降低模型计算成本和参数量的同时，帮助网络学习到更具代表性的特征，从而提升模型性能 。
    * **链接:** [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf) (CVPR 2023)
    
* <u>**TDCNet**</u>
    * **简要说明：** TDCNet 是一种针对移动红外小目标检测的新型网络，旨在解决传统 3D 卷积对运动感知不足的问题。它提出了两个核心即插即用模块：TDCR (时间差分卷积重参数化) 模块，通过重参数化技术将多尺度时间差分操作融合进 3D 卷积，实现了零推理成本的显式运动建模；TDCSTA (TDC引导的时空注意力) 模块，利用 TDCR 提取的强运动特征作为 Query，指导并增强时空特征的语义表达，有效抑制了复杂背景干扰。
    * **链接:** [[Paper]](https://arxiv.org/abs/2511.09352) [[Code]](https://github.com/IVPLaboratory/TDCNet) (AAAI 2026)



---

### <a id="03_Re_Parameterization"></a>📂 03_Re_Parameterization (重参数化)

本章节汇总了“重参数化”技术的论文，这些技术在训练时使用复杂结构，在推理时等效融合为简单结构，以实现性能和速度的平衡。
* *(待补充...)*

---

### <a id="04_Transformer"></a>📂 04_Transformer (Transformer系列)

本章节汇总了基于Transformer架构的视觉模型和模块。

* <u>**AST-Transformer**</u>
    * **简要说明:** 全称为“Adapt or Perish: Adaptive Sparse Transformer”，用于图像恢复。它提出“自适应稀疏Transformer”（AST），使用自适应稀疏自注意力（ASSA）块来过滤掉无关区域的噪声干扰，并使用特征精炼前馈网络（FRFN）来减少特征冗余。
    * **链接:** [[Paper]](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.html) [[Code]](https://github.com/joshyZhou/AST) (CVPR 2024)

* <u>**SwiftFormer**</u>
    * **简要说明:** 一种为移动端设计的实时、高效Transformer模型。它提出“高效加法注意力”，用线性的逐元素乘法取代了昂贵的矩阵乘法，使得注意力模块可以用在网络的所有阶段，实现了极佳的速度-精度权衡。
    * **链接:** [[Paper]](https://arxiv.org/abs/2303.15446) [[Code]](https://github.com/Amshaker/SwiftFormer) (ICCV 2023)

* <u>**WPFormer**</u>
    * **简要说明:** 一种用于像素级表面缺陷检测的查询式Transformer。它提出了“小波和原型增强查询”（Wavelet and Prototype Augmented Query），利用双域（频域和时域）解码器和小波增强通道注意力（WCA）来使模型更关注高频细节，有效检测工业表面的微小缺陷。
    * **链接:** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html) [[Code]](https://github.com/iefengyan/WPFormer) (CVPR 2025)

* <u>**DAT-Transformer**</u>
    * **简要说明:** 提出“可变形注意力Transformer”（Deformable-Attention-based Transformer）。其核心是可变形的自注意力模块，其中“键”和“值”对的位置是根据数据动态学习的，这使得模型能够自适应地聚焦于相关的图像区域，捕获更多信息特征。
    * **链接:** [[Paper]](https://arxiv.org/abs/2308.03364) [[Code]](https://github.com/zhengchen1999/DAT) (ICCV 2023)

* <u>**TBSN**</u>
    * **简要说明:**这篇论文提出了一种名为 **TBSN** 的新型自监督去噪网络 。它通过巧妙地**重新设计 Transformer 的注意力机制**（G-CSA 和 M-WSA），使其能够**在满足“盲点”约束**（BSN）的前提下工作 。这样，模型既能利用 Transformer 强大的全局感知能力，又不会“偷看”到原始噪声，从而在自监督去噪任务上达到了 SOTA 性能。
    * **链接:** [[Paper]](https://arxiv.org/abs/2404.07846) [[Code]](https://github.com/nagejacob/TBSN) (AAAI 2025)

* <u>**SST**</u>
    * **简要说明:**SST (State Space Transformer) 是一种用于时间序列预测的新型混合架构，旨在解决简单堆叠 Mamba 和 Transformer 所导致的“信息干扰”问题。其核心思想是利用多尺度专家（MoE）系统：使用 Mamba 专家在低分辨率（粗粒度）数据上捕捉长程模式，同时使用 Transformer 专家在 high-resolution（细粒度）数据上捕捉短程变异。通过这种方式，SST 在保持线性计算复杂度的同时，实现了最先进的（SOTA）预测性能。
    * **链接:** [[Paper]](https://arxiv.org/abs/2404.14757 ) [[Code]](https://github.com/XiongxiaoXu/SST) (CIKM 2025)https://arxiv.org/abs/2504.04701

* <u>**DFormerv2**</u>
    * **简要说明:**DFormerv2 是一种新颖的 RGB-D 语义分割主干网络，通过将深度信息转化为几何先验（Geometry Prior），并将其融入自注意力机制（Geometry Self-Attention, GSA），实现了对 RGB 和深度信息的有效融合。不同于以往使用双编码器或统一编码器处理深度信息的方法，DFormerv2 无需额外的深度编码层，直接利用深度图中的几何信息指导注意力权重的分配。实验表明，该方法在 NYU Depth V2、SUN RGB-D 等数据集上取得了最先进的性能，且计算成本显著低于现有方法 。
    * **链接:** [[Paper]](https://arxiv.org/abs/2504.04701 ) [[Code]](https://github.com/VCIP-RGBD/DFormer) (CVPR 2025)



---

### <a id="05_Feature_Fusion_Neck"></a>📂 05_Feature_Fusion_Neck (特征融合/Neck系列)

本章节汇总了FPN及其他用于融合多尺度特征的Neck结构创新。

* <u>**ConDSeg**</u>
    * **简要说明:**ConDSeg 是一种针对医学图像分割中“软边界”模糊和“共现现象”误导挑战的通用框架 1111。该方法通过 **语义信息解耦（SID）** 将特征分离为前景、背景和不确定区域，并利用 **对比驱动特征聚合（CDFA）** 模块以对比的方式引导多级特征融合，从而在低对比度环境下实现精准的边界分割 2。配合 **尺寸感知解码器（SA-Decoder）** 对不同尺度目标的分治处理，ConDSeg 有效避免了特征混淆，在多个医学数据集上取得了 SOTA 性能 3。
    * **链接:** [[Paper]](https://arxiv.org/abs/2412.08345 ) [[Code]](https://github.com/Mengqi-Lei/ConDSeg) (AAAI 2025)

* <u>**GLVMamba**</u>
    * **简要说明:**GLVMamba 是一种针对遥感图像分割挑战（如光照阴影、目标尺度差异大）提出的新型网络。它包含两个核心即插即用模块：GLVSS 块 通过引入局部前馈和移位窗口机制，弥补了 Mamba 在保留局部细节上的不足，实现了全局与局部特征的有效融合；SCPP 模块 通过自适应加权的多尺度池化，动态感知不同尺度的目标，有效解决了分割中的空洞和误检问题。这两个模块共同助力 GLVMamba 在遥感图像分割任务上取得了 SOTA 性能。
    * **链接:** [[Paper]](https://arxiv.org/search/?query=GLVMamba%3A+A+Global%E2%80%93Local+Visual+State-Space+Model+for+Remote+Sensing+Image+Segmentation&searchtype=all&source=header](https://arxiv.org/search/?query=GLVMamba%3A+A+Global%E2%80%93Local+Visual+State-Space+Model+for+Remote+Sensing+Image+Segmentation&searchtype=all&source=header ) [[Code]](https://github.com/Tokisakiwlp/GLVMamba) (TGRS 2025)





---

### <a id="06_Frequency_Spatial"></a>📂 06_Frequency_Spatial (频域与空域转换系列)

本章节汇总了结合傅里叶变换等频域处理与空域信息的模型。

* <u>**ASCNet**</u>
    * **简要说明:** 提出“注意力扩张空间光谱卷积网络”，用于高光谱图像（HSI）分类。该模型包含一个用于学习空间特征的注意力CNN分支和一个用于学习光谱相关性的注意力RNN分支，最后将两者联合学习。
    * **链接:** [[Paper]](https://arxiv.org/abs/2401.15578) [[Code]](https://github.com/xdFai/ASCNet) (CVPR 2025)

---

### <a id="07_Activation_Normalization"></a>📂 07_Activation_Normalization (激活函数与归一化系列)

本章节汇总了对ReLU、BatchNorm等模块的改进。
* <u>**MONA**</u>
  * **简要说明:** 一种名为 SCSA (空间与通道协同注意力) 的即插即用注意力模块。它旨在探索空间和通道注意力之间被忽视的“协同效应” ，即现有方法未能解决的“多语义差异”问题。它引入了 `SMSA`（共享多语义空间注意力） ，利用多尺度 1D 卷积来提取不同的空间语义信息 ，并以此作为空间先验来“指导” `PCSA`（渐进式通道自注意力） 。PCSA 进而使用通道自注意力来“缓解”不同语义特征间的“差异” ，从而在分类、检测和分割等多种视觉任务上实现更优的特征融合 。。
  * **链接:** [[Paper]](https://arxiv.org/abs/2408.08345) [[Code]](https://github.com/LeiyiHU/mona) (CVPR 2025)

## 📜 许可证 (License)

本项目采用 [MIT License](LICENSE) 开源。
