"""
即插即用模块集合 (Plug-and-Play Modules)
从ConDSeg模型中提取的可复用模块，可以用于不同的backbone架构
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==================== 基础模块 ====================

class CBR(nn.Module):
    """基础卷积块: Conv + BatchNorm + ReLU"""
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, stride=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False, stride=stride),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


class channel_attention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_planes, ratio=16):
        super(channel_attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return x0 * self.sigmoid(out)


class spatial_attention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x0 = x  # [B,C,H,W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x0 * self.sigmoid(x)


# ==================== 特征增强模块 ====================

class dilated_conv(nn.Module):
    """空洞卷积模块 (FEM - Feature Enhancement Module)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = nn.Sequential(CBR(in_c, out_c, kernel_size=1, padding=0), channel_attention(out_c))
        self.c2 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=6, dilation=6), channel_attention(out_c))
        self.c3 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=12, dilation=12), channel_attention(out_c))
        self.c4 = nn.Sequential(CBR(in_c, out_c, kernel_size=(3, 3), padding=18, dilation=18), channel_attention(out_c))
        self.c5 = CBR(out_c * 4, out_c, kernel_size=3, padding=1, act=False)
        self.c6 = CBR(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.sa = spatial_attention()

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        xc = torch.cat([x1, x2, x3, x4], axis=1)
        xc = self.c5(xc)
        xs = self.c6(x)
        x = self.relu(xc + xs)
        x = self.sa(x)
        return x


# ==================== 特征解耦模块 ====================

class DecoupleLayer(nn.Module):
    """特征解耦层：将特征分解为前景、背景和不确定性特征"""
    def __init__(self, in_c=1024, out_c=256):
        super(DecoupleLayer, self).__init__()
        self.cbr_fg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_bg = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )
        self.cbr_uc = nn.Sequential(
            CBR(in_c, 512, kernel_size=3, padding=1),
            CBR(512, out_c, kernel_size=3, padding=1),
            CBR(out_c, out_c, kernel_size=1, padding=0)
        )

    def forward(self, x):
        f_fg = self.cbr_fg(x)  # 前景特征
        f_bg = self.cbr_bg(x)  # 背景特征
        f_uc = self.cbr_uc(x)  # 不确定性特征
        return f_fg, f_bg, f_uc


# ==================== 辅助头模块 ====================

class AuxiliaryHead(nn.Module):
    """辅助预测头：生成前景、背景和不确定性的辅助预测"""
    def __init__(self, in_c):
        super(AuxiliaryHead, self).__init__()
        self.branch_fg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_bg = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.branch_uc = nn.Sequential(
            CBR(in_c, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/8
            CBR(256, 256, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/4
            CBR(256, 128, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1/2
            CBR(128, 64, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 1
            CBR(64, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, f_fg, f_bg, f_uc):
        mask_fg = self.branch_fg(f_fg)
        mask_bg = self.branch_bg(f_bg)
        mask_uc = self.branch_uc(f_uc)
        return mask_fg, mask_bg, mask_uc


# ==================== CDFA核心模块 ====================

class ContrastDrivenFeatureAggregation(nn.Module):
    """
    对比驱动特征聚合模块 (CDFA - Contrast-Driven Feature Aggregation)
    这是核心的即插即用模块，可以用于任何backbone架构
    """
    def __init__(self, in_c, dim, num_heads, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            CBR(in_c, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )
        self.output_cbr = nn.Sequential(
            CBR(dim, dim, kernel_size=3, padding=1),
            CBR(dim, dim, kernel_size=3, padding=1),
        )

    def forward(self, x, fg, bg):
        """
        Args:
            x: 主特征图 [B, C, H, W]
            fg: 前景特征 [B, C, H, W]
            bg: 背景特征 [B, C, H, W]
        Returns:
            out: 增强后的特征 [B, C, H, W]
        """
        x = self.input_cbr(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):

        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):

        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


# ==================== 预处理模块 ====================

class CDFAPreprocess(nn.Module):
    """CDFA预处理模块：调整特征图尺寸"""
    def __init__(self, in_c, out_c, up_scale):
        super().__init__()
        up_times = int(math.log2(up_scale))
        self.preprocess = nn.Sequential()
        self.c1 = CBR(in_c, out_c, kernel_size=3, padding=1)
        for i in range(up_times):
            self.preprocess.add_module(f'up_{i}', nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            self.preprocess.add_module(f'conv_{i}', CBR(out_c, out_c, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.c1(x)
        x = self.preprocess(x)
        return x


# ==================== 测试函数 ====================

def test_cdfa_module():
    """测试CDFA模块"""
    print("=" * 60)
    print("测试 CDFA 模块")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    channels = 128
    height, width = 16, 16
    
    x = torch.randn(batch_size, channels, height, width).to(device)
    fg = torch.randn(batch_size, channels, height, width).to(device)
    bg = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"\n输入形状:")
    print(f"  主特征 x: {x.shape}")
    print(f"  前景特征 fg: {fg.shape}")
    print(f"  背景特征 bg: {bg.shape}")
    
    # 创建CDFA模块
    cdfa = ContrastDrivenFeatureAggregation(
        in_c=channels,
        dim=channels,
        num_heads=4,
        kernel_size=3,
        padding=1,
        stride=1
    ).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = cdfa(x, fg, bg)
    
    print(f"\n输出形状:")
    print(f"  增强特征: {output.shape}")
    print("✓ CDFA模块测试通过!")
    return output


def test_decouple_layer():
    """测试解耦层"""
    print("\n" + "=" * 60)
    print("测试 DecoupleLayer 模块")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    in_channels = 1024
    out_channels = 128
    height, width = 16, 16
    
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\n输入形状: {x.shape}")
    
    # 创建解耦层
    decouple = DecoupleLayer(in_c=in_channels, out_c=out_channels).to(device)
    
    # 前向传播
    with torch.no_grad():
        f_fg, f_bg, f_uc = decouple(x)
    
    print(f"\n输出形状:")
    print(f"  前景特征 f_fg: {f_fg.shape}")
    print(f"  背景特征 f_bg: {f_bg.shape}")
    print(f"  不确定性特征 f_uc: {f_uc.shape}")
    print("✓ DecoupleLayer模块测试通过!")
    return f_fg, f_bg, f_uc


def test_dilated_conv():
    """测试空洞卷积模块"""
    print("\n" + "=" * 60)
    print("测试 dilated_conv 模块")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height, width = 32, 32
    
    x = torch.randn(batch_size, in_channels, height, width).to(device)
    print(f"\n输入形状: {x.shape}")
    
    # 创建空洞卷积模块
    dconv = dilated_conv(in_c=in_channels, out_c=out_channels).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = dconv(x)
    
    print(f"\n输出形状: {output.shape}")
    print("✓ dilated_conv模块测试通过!")
    return output


def test_auxiliary_head():
    """测试辅助头"""
    print("\n" + "=" * 60)
    print("测试 AuxiliaryHead 模块")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    channels = 128
    height, width = 16, 16
    
    f_fg = torch.randn(batch_size, channels, height, width).to(device)
    f_bg = torch.randn(batch_size, channels, height, width).to(device)
    f_uc = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"\n输入形状:")
    print(f"  f_fg: {f_fg.shape}")
    print(f"  f_bg: {f_bg.shape}")
    print(f"  f_uc: {f_uc.shape}")
    
    # 创建辅助头
    aux_head = AuxiliaryHead(in_c=channels).to(device)
    
    # 前向传播
    with torch.no_grad():
        mask_fg, mask_bg, mask_uc = aux_head(f_fg, f_bg, f_uc)
    
    print(f"\n输出形状:")
    print(f"  前景mask: {mask_fg.shape}")
    print(f"  背景mask: {mask_bg.shape}")
    print(f"  不确定性mask: {mask_uc.shape}")
    print("✓ AuxiliaryHead模块测试通过!")
    return mask_fg, mask_bg, mask_uc


def test_attention_modules():
    """测试注意力机制模块"""
    print("\n" + "=" * 60)
    print("测试 注意力机制 模块")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    channels = 128
    height, width = 32, 32
    
    x = torch.randn(batch_size, channels, height, width).to(device)
    print(f"\n输入形状: {x.shape}")
    
    # 测试通道注意力
    ca = channel_attention(in_planes=channels).to(device)
    with torch.no_grad():
        x_ca = ca(x)
    print(f"通道注意力输出: {x_ca.shape}")
    
    # 测试空间注意力
    sa = spatial_attention(kernel_size=7).to(device)
    with torch.no_grad():
        x_sa = sa(x)
    print(f"空间注意力输出: {x_sa.shape}")
    
    print("✓ 注意力机制模块测试通过!")


def test_integration():
    """集成测试：模拟完整的特征处理流程"""
    print("\n" + "=" * 60)
    print("集成测试：完整特征处理流程")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟backbone输出的特征
    batch_size = 1
    x4 = torch.randn(batch_size, 1024, 16, 16).to(device)  # 最后一层特征
    x3 = torch.randn(batch_size, 512, 32, 32).to(device)   # 倒数第二层
    x2 = torch.randn(batch_size, 256, 64, 64).to(device)   # 第三层
    x1 = torch.randn(batch_size, 64, 128, 128).to(device)  # 第一层
    
    print(f"\n输入特征层级:")
    print(f"  x1: {x1.shape}")
    print(f"  x2: {x2.shape}")
    print(f"  x3: {x3.shape}")
    print(f"  x4: {x4.shape}")
    
    # 1. 特征解耦
    decouple = DecoupleLayer(in_c=1024, out_c=128).to(device)
    f_fg, f_bg, f_uc = decouple(x4)
    print(f"\n1. 解耦后:")
    print(f"  f_fg: {f_fg.shape}, f_bg: {f_bg.shape}, f_uc: {f_uc.shape}")
    
    # 2. 预处理特征（调整到不同层级）
    preprocess_fg3 = CDFAPreprocess(128, 128, 2).to(device)  # 放大2倍
    preprocess_bg3 = CDFAPreprocess(128, 128, 2).to(device)
    f_fg3 = preprocess_fg3(f_fg)
    f_bg3 = preprocess_bg3(f_bg)
    print(f"\n2. 预处理后 (放大到x3层级):")
    print(f"  f_fg3: {f_fg3.shape}, f_bg3: {f_bg3.shape}")
    
    # 3. 空洞卷积特征增强
    dconv3 = dilated_conv(512, 128).to(device)
    d3 = dconv3(x3)
    print(f"\n3. 特征增强后:")
    print(f"  d3: {d3.shape}")
    
    # 4. CDFA特征聚合
    cdfa3 = ContrastDrivenFeatureAggregation(128, 128, 4).to(device)
    f3 = cdfa3(d3, f_fg3, f_bg3)
    print(f"\n4. CDFA聚合后:")
    print(f"  f3: {f3.shape}")
    
    # 5. 辅助头预测
    aux_head = AuxiliaryHead(128).to(device)
    mask_fg, mask_bg, mask_uc = aux_head(f_fg, f_bg, f_uc)
    print(f"\n5. 辅助预测:")
    print(f"  mask_fg: {mask_fg.shape}")
    print(f"  mask_bg: {mask_bg.shape}")
    print(f"  mask_uc: {mask_uc.shape}")
    
    print("\n" + "=" * 60)
    print("✓ 集成测试通过! 所有即插即用模块工作正常")
    print("=" * 60)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "即插即用模块测试" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    try:
        # 测试各个模块
        test_cdfa_module()
        test_decouple_layer()
        test_dilated_conv()
        test_auxiliary_head()
        test_attention_modules()
        test_integration()
        
        print("\n" + "=" * 60)
        print("所有测试完成! ✓")
        print("=" * 60)
        print("\n这些模块可以即插即用地用于不同的backbone架构:")
        print("  - ResNet (network/model.py)")
        print("  - PVTv2 (network_pvt/model.py)")
        print("  - 或其他自定义backbone")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

