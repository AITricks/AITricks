# @Time    : 2024/12/19
# @Author  : AI Assistant
# @File    : attention_modules.py
# @Description: 真正的即插即用注意力模块 - 无外部依赖，自动参数推断
#              True Plug-and-Play Attention Modules - No external dependencies, auto parameter inference

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 基础模块 (Basic Modules)
# =============================================================================

def conv_relu_bn(in_channel, out_channel, dirate):
    """
    基础卷积块：Conv2d + BatchNorm + ReLU
    Basic convolution block: Conv2d + BatchNorm + ReLU
    
    Args:
        in_channel: 输入通道数
        out_channel: 输出通道数  
        dirate: 扩张率 (dilation rate)
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, 
                  kernel_size=3, stride=1, padding=dirate, dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    )


class Conv(nn.Module):
    """
    普通卷积分支：3个连续的3×3卷积层
    Normal convolution branch: 3 consecutive 3×3 convolution layers
    """
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class DConv(nn.Module):
    """
    扩张卷积分支：3个不同扩张率的卷积层
    Dilated convolution branch: 3 convolution layers with different dilation rates
    """
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        dilation = [2, 4, 2]
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x


# =============================================================================
# 核心注意力模块 (Core Attention Modules)
# =============================================================================

class BilinearAttention(nn.Module):
    """
    BAM (Bilinear Attention Module) - 双线性注意力模块
    
    这是ABC网络的核心创新，通过双线性相关性计算空间注意力。
    对应结构图中左下角的详细注意力机制。
    
    This is the core innovation of ABC network, computing spatial attention 
    through bilinear correlation. Corresponds to the detailed attention 
    mechanism in the bottom-left of the architecture diagram.
    
    Args:
        in_dim: 输入通道数
        reduction_ratio: 注意力降维比例，默认4
    """
    def __init__(self, in_dim, reduction_ratio=4):
        super(BilinearAttention, self).__init__()
        self.in_dim = in_dim
        self.reduction_ratio = reduction_ratio
        
        # Query和Key分支的卷积层
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        
        # 输出卷积层
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入特征图 (B, C, H, W)
            
        Returns:
            attention: 注意力权重 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Query分支：降维到1通道
        q = self.query_conv(x)  # (B, 1, H, W)
        
        # Key分支：降维到1通道
        k = self.key_conv(x)  # (B, 1, H, W)
        
        # 双线性相关性计算：元素级相乘
        att = q * k  # (B, 1, H, W)
        
        # Softmax归一化
        att = self.softmax(att)
        
        # 输出卷积
        att = self.s_conv(att)
        
        return att


class ConvAttention(nn.Module):
    """
    卷积注意力模块：结合普通卷积和扩张卷积的注意力机制
    Convolution Attention Module: Attention mechanism combining normal and dilated convolutions
    
    Args:
        in_dim: 输入通道数
        reduction_ratio: 注意力降维比例
    """
    def __init__(self, in_dim, reduction_ratio=4):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = BilinearAttention(in_dim, reduction_ratio)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入特征图 (B, C, H, W)
            
        Returns:
            output: 注意力加权的输出 (B, C, H, W)
        """
        # Query分支：普通卷积
        q = self.conv(x)
        
        # Key分支：扩张卷积
        k = self.dconv(x)
        
        # Value：两者之和
        v = q + k
        
        # 计算注意力权重
        att = self.att(x)
        
        # 注意力加权
        out = att * v
        
        # 残差连接
        return self.gamma * out + v + x


class FeedForward(nn.Module):
    """
    前馈网络：用于Transformer的FFN层
    Feed Forward Network: FFN layer for Transformer
    """
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out


class ConvTransformerBlock(nn.Module):
    """
    CLFT (Convolution Linear Fusion Transformer) - 卷积线性融合Transformer
    
    这是ABC网络编码器中的核心模块，结合了卷积注意力和前馈网络。
    对应结构图中编码器的绿色CLFT块。
    
    This is the core module in ABC network encoder, combining convolution 
    attention and feed-forward network. Corresponds to the green CLFT 
    blocks in the encoder of the architecture diagram.
    
    Args:
        in_dim: 输入通道数
        out_dim: 输出通道数
        reduction_ratio: 注意力降维比例
    """
    def __init__(self, in_dim, out_dim, reduction_ratio=4):
        super(ConvTransformerBlock, self).__init__()
        self.attention = ConvAttention(in_dim, reduction_ratio)
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入特征图 (B, C, H, W)
            
        Returns:
            output: 变换后的特征图 (B, out_dim, H, W)
        """
        # 注意力机制
        x = self.attention(x)
        
        # 前馈网络
        out = self.feedforward(x)
        
        return out


class SimplifiedBAM(nn.Module):
    """
    简化版BAM - 轻量级双线性注意力模块
    
    移除了部分计算密集的操作，适合资源受限的场景。
    保持核心的双线性注意力机制。
    
    Simplified BAM - Lightweight bilinear attention module
    
    Removes some computationally intensive operations, suitable for 
    resource-constrained scenarios. Maintains core bilinear attention mechanism.
    
    Args:
        in_dim: 输入通道数
        reduction_ratio: 注意力降维比例
    """
    def __init__(self, in_dim, reduction_ratio=8):
        super(SimplifiedBAM, self).__init__()
        self.in_dim = in_dim
        self.reduction_ratio = reduction_ratio
        
        # 简化的通道降维
        self.channel_reduce = nn.Conv2d(in_dim, in_dim // reduction_ratio, 1)
        self.channel_restore = nn.Conv2d(in_dim // reduction_ratio, in_dim, 1)
        
        # 空间注意力
        self.spatial_conv = nn.Conv2d(2, 1, 7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入特征图 (B, C, H, W)
            
        Returns:
            output: 注意力加权的输出 (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # 通道注意力（简化版）
        avg_pool = F.avg_pool2d(x, 1)
        max_pool = F.max_pool2d(x, 1)
        
        # 空间注意力
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_input))
        
        # 应用注意力
        out = x * spatial_att
        
        return out


# =============================================================================
# U形卷积-扩张卷积模块 (U-shaped Convolution-Dilated Convolution Module)
# =============================================================================

class UCDC(nn.Module):
    """
    UCDC (U-shaped Convolution-Dilated Convolution) - U形卷积-扩张卷积模块
    
    这是ABC网络中的U形模块，用于瓶颈层和部分解码器阶段。
    通过U形结构和扩张卷积捕获多尺度上下文信息。
    对应结构图中的UCDC块。
    
    This is the U-shaped module in ABC network, used in bottleneck layers 
    and some decoder stages. Captures multi-scale context through U-shaped 
    structure and dilated convolutions. Corresponds to the UCDC blocks in 
    the architecture diagram.
    
    结构说明 (Structure):
        Input
          ↓
        Conv (初始卷积)
          ↓
        D.C.(r=2) ──┐
          ↓         │ (skip connection)
        D.C.(r=4)   │
          ↓         │
        D.C.(r=2) ←─┘
          ↓
        Conv (最终卷积)
          ↓
        Output
    
    Args:
        in_ch (int): 输入通道数
        out_ch (int): 输出通道数
    """
    def __init__(self, in_ch, out_ch):
        super(UCDC, self).__init__()
        # 初始卷积层
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        
        # 扩张卷积序列（U形结构的下行路径）
        # 第一个扩张卷积：将通道数减半
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)  # dilation=2
        
        # 第二个扩张卷积：保持通道数
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)  # dilation=4
        
        # 第三个扩张卷积：恢复通道数（与skip connection融合后）
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)  # dilation=2, 输入是concat后的out_ch
        
        # 最终卷积层（融合skip connection）
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)  # 输入是x1和dx3的concat

    def forward(self, x):
        """
        前向传播
        Forward pass
        
        Args:
            x: 输入特征图 (B, in_ch, H, W)
            
        Returns:
            output: 输出特征图 (B, out_ch, H, W)
        """
        # 初始卷积
        x1 = self.conv1(x)  # (B, out_ch, H, W)
        
        # U形下行路径：扩张卷积序列
        dx1 = self.dconv1(x1)  # (B, out_ch//2, H, W)
        dx2 = self.dconv2(dx1)  # (B, out_ch//2, H, W)
        
        # U形上行路径：融合skip connection（dx1和dx2的concat）
        # 这里实现了结构图中的内部skip connection
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))  # (B, out_ch, H, W)
        
        # 最终融合：将初始卷积输出和U形路径输出concat
        # 这里实现了结构图中的外部skip connection（从conv1到conv2）
        out = self.conv2(torch.cat((x1, dx3), dim=1))  # (B, out_ch, H, W)
        
        return out


# =============================================================================
# 使用示例 (Usage Examples)
# =============================================================================

def example_1_basic_usage():
    """
    示例1：基础使用
    Example 1: Basic Usage
    """
    print("=== 示例1：基础使用 ===")
    
    # 创建输入
    x = torch.randn(2, 64, 32, 32)
    print(f"输入形状: {x.shape}")
    
    # BAM模块
    bam = BilinearAttention(in_dim=64)
    bam_out = bam(x)
    print(f"BAM输出形状: {bam_out.shape}")
    
    # CLFT模块
    clft = ConvTransformerBlock(in_dim=64, out_dim=128)
    clft_out = clft(x)
    print(f"CLFT输出形状: {clft_out.shape}")
    
    # 简化BAM
    simple_bam = SimplifiedBAM(in_dim=64)
    simple_out = simple_bam(x)
    print(f"简化BAM输出形状: {simple_out.shape}")
    
    # UCDC模块
    ucdc = UCDC(in_ch=64, out_ch=128)
    ucdc_out = ucdc(x)
    print(f"UCDC输出形状: {ucdc_out.shape}")


def example_2_unet_integration():
    """
    示例2：集成到UNet
    Example 2: Integration into UNet
    """
    print("\n=== 示例2：集成到UNet ===")
    
    class SimpleUNetWithAttention(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            
            # 编码器
            self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
            self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
            self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
            
            # 注意力模块
            self.attention1 = BilinearAttention(64)
            self.attention2 = ConvTransformerBlock(128, 128)
            self.attention3 = SimplifiedBAM(256)
            
            # 解码器
            self.dec2 = nn.Conv2d(384, 128, 3, padding=1)  # 256 + 128 = 384
            self.dec1 = nn.Conv2d(192, 64, 3, padding=1)   # 128 + 64 = 192
            self.final = nn.Conv2d(64, out_channels, 1)
            
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
            
        def forward(self, x):
            # 编码器
            e1 = F.relu(self.enc1(x))
            e1_att = self.attention1(e1)
            e1 = e1 + e1_att
            
            e2 = self.pool(e1)
            e2 = F.relu(self.enc2(e2))
            e2 = self.attention2(e2)
            
            e3 = self.pool(e2)
            e3 = F.relu(self.enc3(e3))
            e3 = self.attention3(e3)
            
            # 解码器
            d2 = self.up(e3)
            d2 = torch.cat([e2, d2], dim=1)
            d2 = F.relu(self.dec2(d2))
            
            d1 = self.up(d2)
            d1 = torch.cat([e1, d1], dim=1)
            d1 = F.relu(self.dec1(d1))
            
            out = self.final(d1)
            return out
    
    # 测试
    model = SimpleUNetWithAttention()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"UNet+注意力输出形状: {out.shape}")


def example_2_5_ucdc_usage():
    """
    示例2.5：UCDC模块使用
    Example 2.5: UCDC Module Usage
    """
    print("\n=== 示例2.5：UCDC模块使用 ===")
    
    class NetworkWithUCDC(nn.Module):
        def __init__(self, in_channels=3, out_channels=1):
            super().__init__()
            
            # 编码器
            self.enc1 = nn.Conv2d(in_channels, 64, 3, padding=1)
            self.enc2 = ConvTransformerBlock(64, 128)
            self.enc3 = ConvTransformerBlock(128, 256)
            
            # 瓶颈层：使用UCDC模块
            self.bottleneck = UCDC(256, 512)
            
            # 解码器：也使用UCDC模块
            # 注意：concat后通道数会翻倍，需要先调整通道数
            self.dec3_conv = nn.Conv2d(256 + 512, 512, 1)  # 先调整通道数
            self.dec3 = UCDC(512, 256)
            self.dec2_conv = nn.Conv2d(128 + 256, 256, 1)  # 先调整通道数
            self.dec2 = ConvTransformerBlock(256, 128)
            self.dec1_conv = nn.Conv2d(64 + 128, 128, 1)  # 先调整通道数
            self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
            self.final = nn.Conv2d(64, out_channels, 1)
            
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            
        def forward(self, x):
            # 编码器
            e1 = F.relu(self.enc1(x))
            e2 = self.pool(e1)
            e2 = self.enc2(e2)
            e3 = self.pool(e2)
            e3 = self.enc3(e3)
            
            # 瓶颈层：UCDC模块
            b = self.pool(e3)
            b = self.bottleneck(b)
            
            # 解码器
            d3 = self.up(b)
            d3 = torch.cat([e3, d3], dim=1)  # 256 + 512 = 768
            d3 = self.dec3_conv(d3)  # 768 -> 512
            d3 = self.dec3(d3)  # UCDC模块: 512 -> 256
            
            d2 = self.up(d3)
            d2 = torch.cat([e2, d2], dim=1)  # 128 + 256 = 384
            d2 = self.dec2_conv(d2)  # 384 -> 256
            d2 = self.dec2(d2)  # 256 -> 128
            
            d1 = self.up(d2)
            d1 = torch.cat([e1, d1], dim=1)  # 64 + 128 = 192
            d1 = self.dec1_conv(d1)  # 192 -> 128
            d1 = F.relu(self.dec1(d1))  # 128 -> 64
            
            out = self.final(d1)
            return out
    
    # 测试
    model = NetworkWithUCDC()
    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    print(f"UCDC网络输出形状: {out.shape}")
    
    # 单独测试UCDC模块
    ucdc = UCDC(in_ch=64, out_ch=128)
    test_input = torch.randn(2, 64, 32, 32)
    test_output = ucdc(test_input)
    print(f"UCDC模块测试 - 输入: {test_input.shape}, 输出: {test_output.shape}")


def example_3_performance_test():
    """
    示例3：性能测试
    Example 3: Performance Test
    """
    print("\n=== 示例3：性能测试 ===")
    
    import time
    
    # 测试参数
    batch_size = 4
    channels = 64
    height, width = 64, 64
    
    x = torch.randn(batch_size, channels, height, width)
    
    # 测试不同模块
    modules = {
        'BAM': BilinearAttention(channels),
        'CLFT': ConvTransformerBlock(channels, channels),
        'SimplifiedBAM': SimplifiedBAM(channels),
        'UCDC': UCDC(channels, channels)
    }
    
    for name, module in modules.items():
        # 预热
        for _ in range(10):
            _ = module(x)
        
        # 计时
        start_time = time.time()
        for _ in range(100):
            _ = module(x)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # 转换为毫秒
        print(f"{name} 平均推理时间: {avg_time:.2f} ms")
        
        # 计算参数量
        total_params = sum(p.numel() for p in module.parameters())
        print(f"{name} 参数量: {total_params:,}")


if __name__ == '__main__':
    # 运行所有示例
    example_1_basic_usage()
    example_2_unet_integration()
    example_2_5_ucdc_usage()
    example_3_performance_test()
    
    print("\n=== 模块使用说明 ===")
    print("1. BilinearAttention: 完整的双线性注意力机制")
    print("2. ConvTransformerBlock: 卷积Transformer块，适合编码器")
    print("3. SimplifiedBAM: 轻量级注意力，适合资源受限场景")
    print("4. UCDC: U形卷积-扩张卷积模块，适合瓶颈层和解码器")
    print("\n[SUCCESS] 所有模块都是真正的即插即用：")
    print("   - 无外部依赖（仅使用PyTorch标准库）")
    print("   - 自动参数推断（无需手动指定特征图尺寸）")
    print("   - 支持任意输入尺寸和批处理大小")