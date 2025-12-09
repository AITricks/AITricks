"""
即插即用模块集合 (Plug-and-Play Modules)
从CTO架构中提取的可独立使用的模块

包含以下模块：
1. Res2Net Bottle2neck - 多尺度特征提取模块
2. Stitch Attention - 多尺度注意力机制
3. Position Attention Module (PAM) - 位置注意力模块
4. Channel Attention Module (CAM) - 通道注意力模块
5. Dual Attention Head - 双注意力头
6. Boundary Enhancement Module (EAM) - 边界增强模块
7. Boundary Injection Module (DM/BIM) - 边界注入模块
8. Sobel边界检测算子 - 边界提取
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import itertools
from math import log


# ==================== 基础模块 ====================

class ConvBNR(nn.Module):
    """卷积+BN+ReLU基础模块"""
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    """1x1卷积+BN+ReLU模块"""
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BasicConv2d(nn.Module):
    """基础2D卷积模块"""
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


# ==================== Res2Net模块 ====================

class Res2NetBottle2neck(nn.Module):
    """
    Res2Net Bottle2neck模块 (即插即用)
    对应结构图中的Fig.3: Basic module of Res2Net
    
    特点：
    - 多尺度特征提取
    - 层次化的残差连接
    - 可配置的scale参数
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """
        Args:
            inplanes: 输入通道数
            planes: 输出通道数
            stride: 卷积步长
            downsample: 下采样模块
            baseWidth: 基础宽度
            scale: 尺度数量 (对应X1, X2, X3, X4)
            stype: 'normal' 或 'stage'
        """
        super(Res2NetBottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        # 第一个1x1卷积
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        
        # 多个3x3卷积 (对应X2, X3, X4路径)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        # 最后一个1x1卷积
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        # 第一个1x1卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 特征分割 (对应X1, X2, X3, X4)
        spx = torch.split(out, self.width, 1)
        
        # 层次化处理
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]  # 残差连接
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        
        # 处理最后一个分割
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        # 最后一个1x1卷积
        out = self.conv3(out)
        out = self.bn3(out)

        # 全局残差连接
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# ==================== Stitch Attention模块 ====================

def attention(query, key, value):
    """标准注意力计算"""
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    p_attn = F.softmax(scores, dim=-1)
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn


class StitchAttention(nn.Module):
    """
    Stitch Attention模块 (即插即用)
    对应结构图中的Fig.4: Stitch-ViT
    
    特点：
    - 多尺度采样 (stitch rate)
    - 多头注意力机制
    - 可配置的stride参数
    """
    def __init__(self, stride, d_model):
        """
        Args:
            stride: 采样步长列表，例如 [(2,2), (3,3), (4,4)]
            d_model: 特征维度
        """
        super(StitchAttention, self).__init__()
        self.stride = stride
        self.query_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征 [B, C, H, W]
        Returns:
            输出特征 [B, C, H, W]
        """
        b, c, h, w = x.size()
        d_k = c // len(self.stride)
        output = []
        
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        
        for (ws, hs), query, key, value in zip(
            self.stride,
            torch.chunk(_query, len(self.stride), dim=1),
            torch.chunk(_key, len(self.stride), dim=1),
            torch.chunk(_value, len(self.stride), dim=1),
        ):
            out_w, out_h = w // ws, h // hs
            
            # Stitch操作：按stride采样
            query = torch.stack(
                [query[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )  # [B, ws*hs, d_k, out_h, out_w]
            
            # 获取实际的维度
            _, num_patches, d_k_actual, out_h_actual, out_w_actual = query.shape
            query = query.reshape(b, num_patches, d_k_actual * out_w_actual * out_h_actual)

            key = torch.stack(
                [key[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )
            _, num_patches, d_k_actual, out_h_actual, out_w_actual = key.shape
            key = key.reshape(b, num_patches, d_k_actual * out_w_actual * out_h_actual)
            
            value = torch.stack(
                [value[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )
            _, num_patches, d_k_actual, out_h_actual, out_w_actual = value.shape
            value = value.reshape(b, num_patches, d_k_actual * out_w_actual * out_h_actual)

            # 注意力计算
            y, _ = attention(query, key, value)

            # 重塑回原始尺寸
            # y的形状是 [b, ws*hs, d_k*out_w*out_h]
            # 需要reshape成 [b, hs, ws, d_k, out_h, out_w]
            y = y.view(b, ws*hs, d_k_actual, out_h_actual, out_w_actual)
            y = y.view(b, hs, ws, d_k_actual, out_h_actual, out_w_actual)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k_actual, h, w)
            output.append(y)

        # 拼接所有头
        output = torch.cat(output, 1)
        self_attention = self.output_linear(output)

        return self_attention


# ==================== 注意力模块 ====================

class PositionAttentionModule(nn.Module):
    """
    位置注意力模块 (PAM) (即插即用)
    特点：捕获空间位置间的依赖关系
    """
    def __init__(self, in_channels, **kwargs):
        super(PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x
        return out


class ChannelAttentionModule(nn.Module):
    """
    通道注意力模块 (CAM) (即插即用)
    特点：捕获通道间的依赖关系
    """
    def __init__(self, **kwargs):
        super(ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x
        return out


class DualAttentionHead(nn.Module):
    """
    双注意力头模块 (即插即用)
    结合位置注意力和通道注意力
    """
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(DualAttentionHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = PositionAttentionModule(inter_channels, **kwargs)
        self.cam = ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


# ==================== 边界相关模块 ====================

def get_sobel(in_chan, out_chan):
    """
    Sobel边界检测算子 (即插即用)
    用于提取图像边界信息
    """
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y


def run_sobel(conv_x, conv_y, input):
    """运行Sobel算子"""
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input


class BoundaryEnhancementModule(nn.Module):
    """
    边界增强模块 (EAM/BEM) (即插即用)
    对应结构图中的Boundary Enhancement Module
    用于融合多尺度边界信息
    """
    def __init__(self):
        super(BoundaryEnhancementModule, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, x4, x1):
        """
        Args:
            x4: 深层特征 [B, 2048, H/8, W/8]
            x1: 浅层特征 [B, 256, H/4, W/4]
        Returns:
            边界特征图 [B, 1, H/4, W/4]
        """
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)
        return out


class BoundaryInjectionModule(nn.Module):
    """
    边界注入模块 (DM/BIM) (即插即用)
    对应结构图中的Boundary Injection Module
    用于将边界信息注入到解码器特征中
    """
    def __init__(self):
        super(BoundaryInjectionModule, self).__init__()
        self.predict3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, xr, dualattention):
        """
        Args:
            xr: 解码器特征 [B, 64, H, W]
            dualattention: 双注意力特征 [B, 64, H', W']
        Returns:
            注入边界信息后的特征 [B, 1, H, W]
        """
        crop_3 = F.interpolate(dualattention, xr.size()[2:], mode='bilinear', align_corners=False)
        re3_feat = self.predict3(torch.cat([xr, crop_3], dim=1))
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 64, -1, -1).mul(xr)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra3_feat = self.ra2_conv4(x)
        x = ra3_feat + crop_3 + re3_feat
        return x


# ==================== 测试函数 ====================

def test_res2net_bottleneck():
    """测试Res2Net Bottle2neck模块"""
    print("=" * 50)
    print("测试 Res2Net Bottle2neck 模块")
    print("=" * 50)
    
    module = Res2NetBottle2neck(inplanes=256, planes=64, baseWidth=26, scale=4)
    x = torch.randn(2, 256, 64, 64)
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Res2Net Bottle2neck 测试通过\n")


def test_stitch_attention():
    """测试Stitch Attention模块"""
    print("=" * 50)
    print("测试 Stitch Attention 模块")
    print("=" * 50)
    
    # 使用能被所有stride整除的尺寸
    stride = [(2, 2), (4, 4), (8, 8)]
    module = StitchAttention(stride=stride, d_model=256)
    x = torch.randn(2, 256, 64, 64)  # 64能被2, 4, 8整除
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"Stride配置: {stride}")
    print(f"✓ Stitch Attention 测试通过\n")


def test_position_attention():
    """测试位置注意力模块"""
    print("=" * 50)
    print("测试 Position Attention Module 模块")
    print("=" * 50)
    
    module = PositionAttentionModule(in_channels=256)
    x = torch.randn(2, 256, 64, 64)
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Position Attention Module 测试通过\n")


def test_channel_attention():
    """测试通道注意力模块"""
    print("=" * 50)
    print("测试 Channel Attention Module 模块")
    print("=" * 50)
    
    module = ChannelAttentionModule()
    x = torch.randn(2, 256, 64, 64)
    out = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Channel Attention Module 测试通过\n")


def test_dual_attention_head():
    """测试双注意力头模块"""
    print("=" * 50)
    print("测试 Dual Attention Head 模块")
    print("=" * 50)
    
    module = DualAttentionHead(in_channels=256, nclass=1, aux=False)
    x = torch.randn(2, 256, 64, 64)
    outputs = module(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {outputs[0].shape}")
    print(f"✓ Dual Attention Head 测试通过\n")


def test_sobel_operator():
    """测试Sobel边界检测算子"""
    print("=" * 50)
    print("测试 Sobel 边界检测算子")
    print("=" * 50)
    
    sobel_x, sobel_y = get_sobel(3, 1)
    x = torch.randn(2, 3, 256, 256)
    out = run_sobel(sobel_x, sobel_y, x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Sobel 边界检测算子 测试通过\n")


def test_boundary_enhancement():
    """测试边界增强模块"""
    print("=" * 50)
    print("测试 Boundary Enhancement Module 模块")
    print("=" * 50)
    
    module = BoundaryEnhancementModule()
    x1 = torch.randn(2, 256, 64, 64)
    x4 = torch.randn(2, 2048, 8, 8)
    out = module(x4, x1)
    print(f"输入 x1 形状: {x1.shape}")
    print(f"输入 x4 形状: {x4.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Boundary Enhancement Module 测试通过\n")


def test_boundary_injection():
    """测试边界注入模块"""
    print("=" * 50)
    print("测试 Boundary Injection Module 模块")
    print("=" * 50)
    
    module = BoundaryInjectionModule()
    xr = torch.randn(2, 64, 64, 64)
    dualattention = torch.randn(2, 64, 32, 32)
    out = module(xr, dualattention)
    print(f"输入 xr 形状: {xr.shape}")
    print(f"输入 dualattention 形状: {dualattention.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ Boundary Injection Module 测试通过\n")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("CTO架构即插即用模块测试")
    print("=" * 60 + "\n")
    
    # 设置随机种子
    torch.manual_seed(42)
    
    try:
        # 测试各个模块
        test_res2net_bottleneck()
        test_stitch_attention()
        test_position_attention()
        test_channel_attention()
        test_dual_attention_head()
        test_sobel_operator()
        test_boundary_enhancement()
        test_boundary_injection()
        
        print("=" * 60)
        print("✓ 所有模块测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()