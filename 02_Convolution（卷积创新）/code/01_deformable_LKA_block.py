"""
可变形大核注意力 (Deformable Large Kernel Attention, D-LKA) 模块

这是一个即插即用的注意力模块，可以轻松集成到各种编码器-解码器架构中。
该模块通过可变形卷积实现自适应特征采样，结合大核卷积捕获长距离依赖关系。

主要组件：
1. DeformConv: 可变形卷积层，学习空间偏移量
2. deformable_LKA: 可变形大核注意力核心模块
3. deformable_LKA_Attention: 完整的注意力模块（即插即用）

对应论文结构图中的 "2D D-LKA Block"
"""

import torch
import torch.nn as nn
import torchvision


class DeformConv(nn.Module):
    """
    可变形卷积模块
    
    通过学习的偏移量实现自适应空间采样，能够处理不规则形状和几何变换。
    
    Args:
        in_channels (int): 输入通道数
        groups (int): 分组卷积的组数，通常等于 in_channels (深度可分离卷积)
        kernel_size (tuple): 卷积核大小，默认 (3,3)
        padding (int): 填充大小，默认 1
        stride (int): 步长，默认 1
        dilation (int): 膨胀率，默认 1
        bias (bool): 是否使用偏置，默认 True
    """
    
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()
        
        # 偏移量预测网络：为每个采样点预测 (x, y) 偏移量
        # 输出通道数为 2 * kernel_size[0] * kernel_size[1]
        # 因为每个采样点需要 2 个偏移量 (x, y)，总共有 kernel_size[0] * kernel_size[1] 个采样点
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        # 可变形卷积层：使用预测的偏移量进行自适应采样
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, C, H', W')
        """
        # 预测偏移量
        offsets = self.offset_net(x)
        # 使用偏移量进行可变形卷积
        out = self.deform_conv(x, offsets)
        return out


class DeformConv_3x3(nn.Module):
    """
    可变形卷积模块（固定使用 3x3 卷积预测偏移量）
    
    与 DeformConv 的区别：偏移量预测网络固定使用 3x3 卷积，而不是根据 kernel_size 调整。
    
    Args:
        in_channels (int): 输入通道数
        groups (int): 分组卷积的组数
        kernel_size (tuple): 可变形卷积的核大小
        padding (int): 填充大小
        stride (int): 步长
        dilation (int): 膨胀率
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_3x3, self).__init__()  # 修复：原代码使用了错误的父类名称
        
        # 偏移量预测网络：固定使用 3x3 卷积
        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    bias=True)

        # 可变形卷积层
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, C, H', W')
        """
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out


class DeformConv_experimental(nn.Module):
    """
    实验性可变形卷积模块
    
    使用深度可分离卷积和通道调整来预测偏移量，可能具有更好的参数效率。
    
    Args:
        in_channels (int): 输入通道数
        groups (int): 分组卷积的组数
        kernel_size (tuple): 卷积核大小
        padding (int): 填充大小
        stride (int): 步长
        dilation (int): 膨胀率
        bias (bool): 是否使用偏置
    """
    
    def __init__(self, in_channels, groups, kernel_size=(3,3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv_experimental, self).__init__()
        
        # 通道调整：将输入通道数调整为偏移量所需的通道数
        self.conv_channel_adjust = nn.Conv2d(in_channels=in_channels, 
                                             out_channels=2 * kernel_size[0] * kernel_size[1], 
                                             kernel_size=(1,1))

        # 偏移量预测网络：使用深度可分离卷积（groups = out_channels）
        self.offset_net = nn.Conv2d(in_channels=2 * kernel_size[0] * kernel_size[1],
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=3,
                                    padding=1,
                                    stride=1,
                                    groups=2 * kernel_size[0] * kernel_size[1],  # 深度可分离卷积
                                    bias=True)

        # 可变形卷积层
        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=in_channels,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, C, H', W')
        """
        # 先调整通道数，再预测偏移量
        x_chan = self.conv_channel_adjust(x)
        offsets = self.offset_net(x_chan)
        out = self.deform_conv(x, offsets)
        return out


class deformable_LKA(nn.Module):
    """
    可变形大核注意力核心模块 (Deformable Large Kernel Attention)
    
    这是 D-LKA 的核心组件，通过两个可变形卷积实现大感受野的注意力机制：
    1. 5x5 可变形深度卷积：捕获局部特征
    2. 7x7 可变形深度膨胀卷积 (dilation=3)：捕获长距离依赖（等效感受野约 19x19）
    3. 1x1 卷积：特征融合
    
    对应结构图中的空间门控单元 (Spatial Gating Unit)
    
    Args:
        dim (int): 输入特征维度（通道数）
    """
    
    def __init__(self, dim):
        super().__init__()
        # 5x5 可变形深度卷积：局部特征提取
        self.conv0 = DeformConv(dim, kernel_size=(5,5), padding=2, groups=dim)
        # 7x7 可变形深度膨胀卷积：长距离依赖（dilation=3 使得等效感受野更大）
        self.conv_spatial = DeformConv(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        # 1x1 卷积：特征融合和维度调整
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 注意力加权的特征图，形状为 (B, C, H, W)
        """
        # 保存原始输入用于残差连接（通过逐元素乘法实现注意力机制）
        u = x.clone()
        # 通过可变形卷积生成注意力权重
        attn = self.conv0(x)           # 5x5 可变形卷积
        attn = self.conv_spatial(attn) # 7x7 可变形膨胀卷积
        attn = self.conv1(attn)        # 1x1 卷积融合
        # 返回注意力加权的特征（对应结构图中的逐元素乘法）
        return u * attn


class deformable_LKA_experimental(nn.Module):
    """
    实验性可变形大核注意力模块
    
    使用实验性可变形卷积实现的版本，可能具有不同的参数效率或性能特征。
    
    Args:
        dim (int): 输入特征维度（通道数）
    """
    
    def __init__(self, dim):
        super().__init__()
        self.conv0 = DeformConv_experimental(dim, kernel_size=(5,5), padding=2, groups=dim)
        self.conv_spatial = DeformConv_experimental(dim, kernel_size=(7,7), stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 注意力加权的特征图，形状为 (B, C, H, W)
        """
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return u * attn


class deformable_LKA_Attention(nn.Module):
    """
    可变形大核注意力模块（即插即用）
    
    这是完整的即插即用注意力模块，可以直接替换到任何编码器-解码器架构中。
    结构对应论文中的 "2D D-LKA Attention" 模块：
    - Conv 1x1 (proj_1) -> GELU -> deformable_LKA -> Conv 1x1 (proj_2) -> 残差连接
    
    该模块可以在 U-Net、Transformer 等架构的编码器或解码器中使用。
    
    Args:
        d_model (int): 特征维度（通道数），应与输入特征图的通道数一致
    
    Input:
        x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
    
    Output:
        torch.Tensor: 输出特征图，形状为 (B, C, H, W)，与输入形状相同
    """
    
    def __init__(self, d_model):
        super().__init__()
        # 第一个 1x1 卷积：特征投影
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        # GELU 激活函数
        self.activation = nn.GELU()
        # 可变形大核注意力核心模块（空间门控单元）
        self.spatial_gating_unit = deformable_LKA(d_model)
        # 第二个 1x1 卷积：特征投影
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        """
        前向传播
        
        实现结构图中的 D-LKA Attention 模块：
        x -> proj_1 -> GELU -> deformable_LKA -> proj_2 -> + (残差连接) -> 输出
        
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, C, H, W)
        """
        # 保存输入用于残差连接
        shorcut = x.clone()
        # 特征投影
        x = self.proj_1(x)
        # 激活
        x = self.activation(x)
        # 可变形大核注意力（空间门控）
        x = self.spatial_gating_unit(x)
        # 特征投影
        x = self.proj_2(x)
        # 残差连接
        x = x + shorcut
        return x


class deformable_LKA_Attention_experimental(nn.Module):
    """
    实验性可变形大核注意力模块（即插即用）
    
    使用实验性可变形卷积实现的版本。
    
    Args:
        d_model (int): 特征维度（通道数）
    """
    
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = deformable_LKA_experimental(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 输入特征图，形状为 (B, C, H, W)
        
        Returns:
            torch.Tensor: 输出特征图，形状为 (B, C, H, W)
        """
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x
    
# 可选依赖：用于计算 FLOPs（仅在主函数中使用）
try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False


if __name__ == '__main__':
    print("=" * 60)
    print("Deformable LKA 模块测试")
    print("=" * 60)
    
    # 自动检测设备（GPU 或 CPU）
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cpu':
        print("提示: 检测到使用 CPU，运行速度可能较慢。")
    print()
    
    # 创建输入数据
    input = torch.rand(1, 96, 56, 56).to(device)
    print(f"输入形状: {input.shape}")
    
    # 创建模型
    print("\n创建模型...")
    lka_layer = deformable_LKA_Attention(d_model=96).to(device)
    lka_layer_exp = deformable_LKA_Attention_experimental(d_model=96).to(device)
    
    # 前向传播测试
    print("运行前向传播...")
    output = lka_layer(input)
    output_exp = lka_layer_exp(input)
    
    print(f"输出形状: {output.shape}")
    print(f"输出形状 (experimental): {output_exp.shape}")
    print(f"输入输出形状匹配: {input.shape == output.shape}")
    print()
    
    # 计算参数数量
    n_parameters = sum(p.numel() for p in lka_layer.parameters() if p.requires_grad)
    n_parameters_exp = sum(p.numel() for p in lka_layer_exp.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("模型统计信息")
    print("=" * 60)
    print(f"deformable_LKA_Attention:")
    print(f"  - 可训练参数数量: {round(n_parameters * 1e-3, 4)} K ({n_parameters:,})")
    
    print(f"\ndeformable_LKA_Attention_experimental:")
    print(f"  - 可训练参数数量: {round(n_parameters_exp * 1e-3, 4)} K ({n_parameters_exp:,})")
    
    # 如果安装了 fvcore，计算 FLOPs
    if HAS_FVCORE:
        print("\n计算 FLOPs...")
        try:
            flops = FlopCountAnalysis(lka_layer, input)
            flops_exp = FlopCountAnalysis(lka_layer_exp, input)
            
            model_flops = flops.total()
            model_flops_exp = flops_exp.total()
            
            print(f"deformable_LKA_Attention:")
            print(f"  - MAdds: {round(model_flops * 1e-6, 4)} M")
            print(f"\ndeformable_LKA_Attention_experimental:")
            print(f"  - MAdds: {round(model_flops_exp * 1e-6, 4)} M")
        except Exception as e:
            print(f"FLOPs 计算失败: {e}")
    else:
        print("\n跳过 FLOPs 计算（未安装 fvcore）")
        print("提示: 要计算 FLOPs，请安装 fvcore: pip install fvcore")
    
    print()
    print("=" * 60)
    print("测试完成！")
    print("=" * 60)