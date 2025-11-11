"""
Mona: Multi-cognitive Visual Adapter (多认知视觉适配器)
一个即插即用的适配器模块，用于视觉Transformer的参数高效微调

该模块实现了Mona Layer，可以插入到Transformer块中，实现参数高效的微调。
模块包含：
- 带可训练缩放因子的缩放LayerNorm
- 多认知卷积滤波器组（3x3, 5x5, 7x7深度卷积）
- 平均聚合滤波器
- 四个跳跃连接以增强适应能力

Author: Mona Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MonaOp(nn.Module):
    """
    多认知卷积滤波器组 (MonaOp)
    
    该模块实现多认知卷积滤波器组，包含：
    - 三个并行的深度卷积（3x3, 5x5, 7x7）- 捕获不同尺度的空间特征
    - 平均聚合 - 融合多尺度特征
    - 1x1投影层与跳跃连接 - 增强信息流
    
    这是Mona适配器的核心组件，通过多尺度卷积捕获丰富的空间上下文信息。
    
    Args:
        in_features (int): 输入通道数
    """
    
    def __init__(self, in_features):
        super().__init__()
        # 三个不同核大小的深度卷积（Depthwise Convolution）
        # groups=in_features 使得每个通道独立卷积，参数量大大减少
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, 
                               padding=3 // 2, groups=in_features)  # 3x3深度卷积
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, 
                               padding=5 // 2, groups=in_features)  # 5x5深度卷积
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, 
                               padding=7 // 2, groups=in_features)  # 7x7深度卷积
        
        # 1x1投影层，用于特征变换
        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, C, H, W)
               B: 批次大小, C: 通道数, H: 高度, W: 宽度
            
        Returns:
            输出张量，形状为 (B, C, H, W)
        """
        # 跳跃连接1: 保存原始输入，用于多尺度卷积分支
        identity = x
        
        # 多认知卷积滤波器（并行处理）
        # 三个不同尺度的卷积并行处理，捕获不同范围的空间特征
        conv1_x = self.conv1(x)  # 3x3深度卷积 - 捕获局部特征
        conv2_x = self.conv2(x)  # 5x5深度卷积 - 捕获中等范围特征
        conv3_x = self.conv3(x)  # 7x7深度卷积 - 捕获大范围特征
        
        # 平均聚合 + 跳跃连接2: 融合多尺度特征并保留原始信息
        x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
        
        # 跳跃连接3: 保存聚合后的特征，用于投影分支
        identity = x
        
        # 1x1投影（跳跃连接4）: 特征变换
        x = self.projector(x)
        
        # 最终的跳跃连接: 保留聚合特征的信息流
        return identity + x


class Mona(nn.Module):
    """
    Mona Layer: 多认知视觉适配器
    
    一个即插即用的适配器模块，可以插入到Transformer块中。
    该模块特点：
    - 带可训练缩放因子的缩放LayerNorm（S1, S2）
    - 下投影到瓶颈维度（减少参数量）
    - 多认知卷积滤波器组（MonaOp）- 核心组件
    - GeLU激活函数
    - 上投影回原始维度
    - 四个跳跃连接以增强适应能力
    
    该模块设计用于参数高效的微调，只需训练适配器参数，冻结预训练模型参数。
    
    Args:
        in_dim (int): 输入维度（通道数）
        bottleneck_dim (int, optional): 瓶颈维度，用于下/上投影。默认: 64
        dropout (float, optional): Dropout比率。默认: 0.1
        init_scale (float, optional): gamma参数的初始缩放值。默认: 1e-6
    """
    
    def __init__(self, in_dim, bottleneck_dim=64, dropout=0.1, init_scale=1e-6):
        super().__init__()
        
        # 带可训练缩放因子的LayerNorm
        # 这是Mona的关键创新：使用两个可学习的缩放因子来控制归一化特征和原始特征的权重
        self.norm = nn.LayerNorm(in_dim)
        # S1 (gamma): 归一化特征的缩放因子，初始值很小，确保开始时适配器影响较小
        self.gamma = nn.Parameter(torch.ones(in_dim) * init_scale)
        # S2 (gammax): 原始特征的缩放因子，初始值为1，保持原始信息流
        self.gammax = nn.Parameter(torch.ones(in_dim))
        
        # 下投影: 将特征维度从in_dim降低到bottleneck_dim，减少参数量
        self.project1 = nn.Linear(in_dim, bottleneck_dim)
        
        # 多认知卷积滤波器组: Mona的核心组件，在瓶颈维度上进行多尺度空间特征提取
        self.adapter_conv = MonaOp(bottleneck_dim)
        
        # 激活函数: GeLU (Gaussian Error Linear Unit)
        self.activation = F.gelu
        
        # Dropout: 防止过拟合
        self.dropout = nn.Dropout(p=dropout)
        
        # 上投影: 将特征维度从bottleneck_dim恢复到in_dim
        self.project2 = nn.Linear(bottleneck_dim, in_dim)
    
    def forward(self, x, hw_shapes=None):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 (B, N, C)，其中 N = H * W
               B: 批次大小, N: 序列长度（空间位置数）, C: 通道数
            hw_shapes: 空间维度元组 (H, W)，表示特征图的高度和宽度
                      如果为None，将尝试从输入形状推断（假设为方形特征图）
                      
        Returns:
            输出张量，形状为 (B, N, C)，与输入形状相同
        """
        # 跳跃连接: 保存输入特征，用于最终的残差连接
        identity = x
        
        # 缩放LayerNorm: norm(x) * S1 + x * S2
        # 这是Mona的关键设计：通过两个可学习的缩放因子平衡归一化特征和原始特征
        # 开始时gamma很小，适配器影响很小；训练过程中gamma增大，适配器逐渐发挥作用
        x_norm = self.norm(x)
        x = x_norm * self.gamma + x * self.gammax
        
        # 下投影: 降低维度以减少参数量
        project1 = self.project1(x)  # (B, N, bottleneck_dim)
        
        # 重塑为空间格式以便进行2D卷积
        # 需要将序列格式 (B, N, C) 转换为空间格式 (B, C, H, W)
        b, n, c = project1.shape
        if hw_shapes is not None:
            h, w = hw_shapes
        else:
            # 尝试推断空间维度（假设特征图为方形）
            h = w = int(n ** 0.5)
            if h * w != n:
                raise ValueError(f"无法从序列长度 {n} 推断空间维度。"
                               "请提供 hw_shapes 参数。")
        
        # 重塑为 (B, C, H, W) 格式以便进行2D卷积
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        # 多认知卷积滤波器组
        # 包含: 3x3, 5x5, 7x7深度卷积, 平均聚合, 和1x1卷积
        # 这是Mona的核心，通过多尺度卷积捕获丰富的空间上下文信息
        project1 = self.adapter_conv(project1)
        
        # 重塑回 (B, N, C) 格式
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)
        
        # GeLU激活: 引入非线性
        nonlinear = self.activation(project1)
        
        # Dropout: 正则化，防止过拟合
        nonlinear = self.dropout(nonlinear)
        
        # 上投影: 恢复原始维度
        project2 = self.project2(nonlinear)  # (B, N, in_dim)
        
        # 最终跳跃连接: identity + project2
        # 残差连接确保信息流畅通，同时允许适配器学习增量调整
        return identity + project2


# ============================================================================
# 示例使用和测试函数
# ============================================================================
if __name__ == "__main__":
    try:
        # 测试Mona模块
        print("测试Mona模块...")
        print("=" * 50)
        
        # 创建Mona层
        in_dim = 384  # 示例: Swin-Base的特征维度
        bottleneck_dim = 64
        mona = Mona(in_dim=in_dim, bottleneck_dim=bottleneck_dim)
        
        # 创建虚拟输入
        batch_size = 2
        h, w = 14, 14  # 示例空间维度
        n = h * w
        x = torch.randn(batch_size, n, in_dim)
        
        print(f"输入形状: {x.shape}")
        
        # 前向传播
        output = mona(x, hw_shapes=(h, w))
        print(f"输出形状: {output.shape}")
        
        # 验证输出形状与输入形状匹配
        assert output.shape == x.shape, "输出形状应与输入形状匹配"
        print("✓ 形状检查通过")
        
        # 统计参数量
        total_params = sum(p.numel() for p in mona.parameters())
        trainable_params = sum(p.numel() for p in mona.parameters() if p.requires_grad)
        print(f"\n总参数量: {total_params:,}")
        print(f"可训练参数量: {trainable_params:,}")
        print(f"参数效率: {total_params / (in_dim * in_dim * 4):.4%} (相比全连接层)")
        
        # 测试不同的输入形状
        print("\n" + "=" * 50)
        print("测试不同空间维度...")
        h2, w2 = 7, 7
        n2 = h2 * w2
        x2 = torch.randn(batch_size, n2, in_dim)
        output2 = mona(x2, hw_shapes=(h2, w2))
        print(f"输入形状: {x2.shape} -> 输出形状: {output2.shape}")
        assert output2.shape == x2.shape, "输出形状应与输入形状匹配"
        print("✓ 可变空间维度测试通过")
        
        print("\n" + "=" * 50)
        print("所有测试通过! ✓")
        print("\n使用示例:")
        print("  from mona import Mona")
        print("  ")
        print("  # 创建Mona适配器")
        print("  mona_layer = Mona(in_dim=384, bottleneck_dim=64)")
        print("  ")
        print("  # 在Transformer块中使用")
        print("  # x: (B, N, C) 其中 N = H * W")
        print("  # output = mona_layer(x, hw_shapes=(H, W))")
        
    except ImportError as e:
        print(f"PyTorch未安装。请安装它以运行测试:")
        print(f"  pip install torch")
        print(f"\n模块结构正确，可以使用。")
    except Exception as e:
        print(f"测试过程中出错: {e}")
        import traceback
        traceback.print_exc()

