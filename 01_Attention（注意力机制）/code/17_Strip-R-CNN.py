"""
Strip R-CNN 即插即用模块提取
Plug-and-Play Modules Extracted from Strip R-CNN

本文件包含从Strip R-CNN论文中提取的即插即用模块:
1. StripBlock - 条形卷积注意力模块 (来自Figure 4的Strip Module)
2. CenterPooling - 中心池化空间注意力模块 (来自Figure 3的Spatial Attention)
3. StripAttention - 完整的条形注意力模块 (包含投影层)

这些模块可以直接插入到任何CNN网络中以增强特征提取能力。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StripBlock(nn.Module):
    """
    条形卷积块 - Strip Convolution Block
    
    这是Strip R-CNN的核心即插即用模块,使用水平和垂直条形卷积来捕获长条形目标的特征。
    
    结构 (来自Figure 4):
    - Square Conv (5x5 depthwise conv)
    - H_Strip Conv (1x19 horizontal strip conv)  
    - V_Strip Conv (19x1 vertical strip conv)
    - PW Conv (1x1 pointwise conv)
    - 最后与输入相乘实现注意力机制
    
    Args:
        dim (int): 输入和输出的通道数
        strip_kernel_size (int): 条形卷积核的长边大小,默认19
    """
    def __init__(self, dim, strip_kernel_size=19):
        super().__init__()
        # Square convolution - 初始特征提取
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        
        # Horizontal strip convolution - 水平条形卷积
        self.strip_conv_h = nn.Conv2d(
            dim, dim, 
            kernel_size=(1, strip_kernel_size), 
            stride=1, 
            padding=(0, strip_kernel_size // 2), 
            groups=dim
        )
        
        # Vertical strip convolution - 垂直条形卷积
        self.strip_conv_v = nn.Conv2d(
            dim, dim, 
            kernel_size=(strip_kernel_size, 1), 
            stride=1, 
            padding=(strip_kernel_size // 2, 0), 
            groups=dim
        )
        
        # Pointwise convolution - 1x1卷积融合特征
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            输出特征图 [B, C, H, W] - 经过注意力加权的特征
        """
        u = x.clone()  # 保存输入用于注意力机制
        
        # 生成注意力权重
        attn = self.conv0(x)
        attn = self.strip_conv_h(attn)  # 水平条形卷积
        attn = self.strip_conv_v(attn)  # 垂直条形卷积
        attn = self.conv1(attn)         # 1x1卷积
        
        # 注意力加权
        return u * attn


class CenterPooling(nn.Module):
    """
    中心池化模块 - Center Pooling for Spatial Attention
    
    这是一个空间注意力模块,通过在水平和垂直方向上进行最大池化来捕获空间信息。
    类似于Figure 3中的Spatial Attention机制。
    
    Args:
        in_channels (int): 输入通道数
        mid_channels (int): 中间层通道数
        out_channels (int): 输出通道数
    """
    def __init__(self, in_channels=256, mid_channels=128, out_channels=256):
        super(CenterPooling, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        
        # 水平方向特征提取
        self.conv1 = self._make_conv(in_channels, mid_channels, 3, padding=1)
        # 垂直方向特征提取
        self.conv2 = self._make_conv(in_channels, mid_channels, 3, padding=1)
        # 残差连接
        self.conv3 = self._make_conv(in_channels, out_channels, 1, with_relu=False)
        # 融合特征
        self.conv4 = self._make_conv(mid_channels, out_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def _make_conv(self, in_channels, out_channels, kernel_size, 
                   stride=1, padding=0, with_relu=True):
        """构建卷积层"""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                     stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def forward(self, feature):
        """
        前向传播
        
        Args:
            feature: 输入特征图 [B, C, H, W]
            
        Returns:
            输出特征图 [B, C, H, W] - 经过空间注意力增强的特征
        """
        # 水平方向池化
        x1 = self.conv1(feature)
        x1_pool = torch.max(x1, dim=-1)[0]  # 在宽度维度上最大池化
        x1_pool = x1_pool.unsqueeze(3).expand_as(x1)
        
        # 垂直方向池化
        x2 = self.conv2(feature)
        x2_pool = torch.max(x2, dim=2)[0]  # 在高度维度上最大池化
        x2_pool = x2_pool.unsqueeze(2).expand_as(x2)
        
        # 融合水平和垂直信息
        x_pool = x1_pool + x2_pool
        x = self.conv4(x_pool)
        
        # 残差连接
        x3 = self.conv3(feature)
        x = x + x3
        x = self.relu(x)
        
        return x


class StripAttention(nn.Module):
    """
    完整的条形注意力模块 - Complete Strip Attention Module
    
    这是一个完整的注意力模块,包含投影层和StripBlock。
    对应Figure 4中Strip Block的完整结构。
    
    Args:
        dim (int): 输入输出通道数
        strip_kernel_size (int): 条形卷积核大小
    """
    def __init__(self, dim, strip_kernel_size=19):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.strip_block = StripBlock(dim, strip_kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, 1)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            输出特征图 [B, C, H, W]
        """
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.strip_block(x)
        x = self.proj_2(x)
        x = x + shortcut  # 残差连接
        return x


class StripEnhancedBlock(nn.Module):
    """
    增强的条形卷积块 - Enhanced Strip Block with FFN
    
    这是一个完整的Transformer-like块,包含StripAttention和FFN。
    对应Figure 4中的完整Strip Block结构 (Norm + Strip Block + Norm + FFN)。
    
    Args:
        dim (int): 特征维度
        mlp_ratio (float): FFN扩展比例
        strip_kernel_size (int): 条形卷积核大小
        drop (float): Dropout比例
    """
    def __init__(self, dim, mlp_ratio=4.0, strip_kernel_size=19, drop=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        
        self.strip_attn = StripAttention(dim, strip_kernel_size)
        
        # Feed-Forward Network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, 1),
            nn.Dropout(drop)
        )
        
        # Layer Scale (可学习的缩放因子)
        self.layer_scale_1 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-2)
        self.layer_scale_2 = nn.Parameter(torch.ones(dim, 1, 1) * 1e-2)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            
        Returns:
            输出特征图 [B, C, H, W]
        """
        # Strip Attention分支
        x = x + self.layer_scale_1 * self.strip_attn(self.norm1(x))
        # FFN分支
        x = x + self.layer_scale_2 * self.mlp(self.norm2(x))
        return x


def main():
    """
    简单测试 - 验证模块可以正常运行
    """
    print("Strip R-CNN 即插即用模块测试\n")
    
    # 创建测试输入张量
    batch_size = 2
    channels = 256
    height = 32
    width = 32
    x = torch.randn(batch_size, channels, height, width)
    
    print(f"输入张量: {x.shape}\n")
    
    # 测试 1: StripBlock
    print("测试 StripBlock...")
    strip_block = StripBlock(dim=channels, strip_kernel_size=19)
    out1 = strip_block(x)
    print(f"  输出: {out1.shape} ✓\n")
    
    # 测试 2: CenterPooling
    print("测试 CenterPooling...")
    center_pooling = CenterPooling(in_channels=channels, mid_channels=128, out_channels=channels)
    out2 = center_pooling(x)
    print(f"  输出: {out2.shape} ✓\n")
    
    # 测试 3: StripAttention
    print("测试 StripAttention...")
    strip_attn = StripAttention(dim=channels, strip_kernel_size=19)
    out3 = strip_attn(x)
    print(f"  输出: {out3.shape} ✓\n")
    
    # 测试 4: StripEnhancedBlock
    print("测试 StripEnhancedBlock...")
    strip_enhanced = StripEnhancedBlock(dim=channels, mlp_ratio=4.0, strip_kernel_size=19)
    out4 = strip_enhanced(x)
    print(f"  输出: {out4.shape} ✓\n")
    
    print("=" * 50)
    print("所有模块测试通过! ✓")
    print("=" * 50)


if __name__ == "__main__":
    main()
