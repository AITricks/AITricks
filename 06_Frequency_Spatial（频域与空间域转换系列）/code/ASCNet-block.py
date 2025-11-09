"""
ASCNet 可插拔模块实现
对应论文 Fig. 3 中的三个核心模块：
- (a) Pixel Shuffle (PS): PixelShuffleUpsample
- (b) Residual Haar Discrete Wavelet Transform (RHDWT): RHDWTBlock  
- (c) Column Non-uniformity Correction Module (CNCM): NewBlock (内部包含 RCSSC)

这些模块可以从 ASCNet 中独立使用，便于在其他网络中复用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入小波库；若不可用，则在 RHDWTBlock 中回退为步长卷积的简化替代实现
try:
    from pytorch_wavelets import DWTForward
    _HAS_WAVELETS = True
except Exception:
    DWTForward = None
    _HAS_WAVELETS = False


class ChannelPool(nn.Module):
    """
    通道池化层：将最大池化和平均池化结果拼接
    用于空间注意力模块中压缩通道维度
    """
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Basic(nn.Module):
    """
    基础卷积块：Conv2d + BatchNorm + LeakyReLU
    用于构建其他复杂模块的基础组件
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True, bias=False):
        super(Basic, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CALayer(nn.Module):
    """
    通道注意力层 (Channel Attention Layer)
    用于 RCSSC 模块中，实现列方向的通道注意力机制
    
    实现细节：
    - 使用自适应池化提取列方向（宽度维度）的全局信息
    - 分别对高度和宽度维度进行注意力计算
    - 通过 Sigmoid 生成注意力权重并应用到特征图上
    """
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # 列方向池化：保留宽度维度，压缩高度维度
        self.avgPoolW = nn.AdaptiveAvgPool2d((1, None))
        self.maxPoolW = nn.AdaptiveMaxPool2d((1, None))

        # 融合平均池化和最大池化结果
        self.conv_1x1 = nn.Conv2d(in_channels=2 * channel, out_channels=2 * channel, kernel_size=1, padding=0, stride=1,
                                  bias=False)
        self.bn = nn.BatchNorm2d(2 * channel, eps=1e-5, momentum=0.01, affine=True)
        self.Relu = nn.LeakyReLU()

        # 高度维度的注意力分支（F_h）
        self.F_h = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        # 宽度维度的注意力分支（F_w）
        self.F_w = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.BatchNorm2d(channel // reduction, eps=1e-5, momentum=0.01, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        N, C, H, W = x.size()
        res = x
        # 拼接平均池化和最大池化结果
        x_cat = torch.cat([self.avgPoolW(x), self.maxPoolW(x)], 1)
        x = self.Relu(self.bn(self.conv_1x1(x_cat)))
        # 分离为高度和宽度两个分支
        x_1, x_2 = x.split(C, 1)
        x_1 = self.F_h(x_1)
        x_2 = self.F_w(x_2)
        # 生成注意力权重
        s_h = self.sigmoid(x_1)
        s_w = self.sigmoid(x_2)
        # 应用注意力权重
        out = res * s_h.expand_as(res) * s_w.expand_as(res)
        return out


class spatial_attn_layer(nn.Module):
    """
    空间注意力层 (Spatial Attention Layer)
    用于 RCSSC 模块中，实现空间维度的注意力机制
    
    实现细节：
    - 通过通道池化压缩通道维度（最大池化 + 平均池化）
    - 使用卷积生成空间注意力图
    - 通过 Sigmoid 生成注意力权重并应用到特征图上
    """
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = Basic(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, bn=False, relu=False)

    def forward(self, x):
        # 压缩通道维度：从 C 通道压缩到 2 通道（最大池化 + 平均池化）
        x_compress = self.compress(x)
        # 生成空间注意力图：从 2 通道压缩到 1 通道
        x_out = self.spatial(x_compress)
        # 生成空间注意力权重
        scale = torch.sigmoid(x_out)
        # 应用空间注意力
        return x * scale


class RCSSC(nn.Module):
    """
    残差列空间自校正块 (Residual Column Spatial Self-Correction, RCSSC)
    对应 Fig. 3 (c) CNCM 模块内部的核心组件
    
    该模块由三个分支组成：
    1. 空间注意力分支 (SA): 捕获空间维度的关键信息
    2. 通道注意力分支 (CA): 捕获列方向的通道依赖关系
    3. 低频上下文分支 (SC): 通过池化提取全局上下文信息
    
    输入输出尺寸：保持不变 (C x H x W)
    """
    def __init__(self, n_feat, reduction=16):
        super(RCSSC, self).__init__()
        pooling_r = 4
        
        # 头部卷积：初始特征提取
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.LeakyReLU(),
        )
        
        # 低频上下文分支 (SC): 通过池化提取全局上下文
        # 对应 Fig. 3 (c) 中的池化分支
        self.SC = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),  # 下采样提取低频信息
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(n_feat)
        )
        
        # 空间注意力分支 (SA)
        self.SA = spatial_attn_layer()
        
        # 通道注意力分支 (CA)
        self.CA = CALayer(n_feat, reduction)

        # 融合空间注意力和通道注意力的分支
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(n_feat * 2, n_feat, kernel_size=1),  # 压缩拼接后的特征
            nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1, stride=1, bias=True)
        )
        self.ReLU = nn.LeakyReLU()
        
        # 尾部卷积：最终特征融合
        self.tail = nn.Conv2d(in_channels=n_feat, out_channels=n_feat, kernel_size=3, padding=1)

    def forward(self, x):
        res = x  # 残差连接
        
        # 1. 头部特征提取
        x = self.head(x)
        
        # 2. 空间注意力分支
        sa_branch = self.SA(x)
        
        # 3. 通道注意力分支
        ca_branch = self.CA(x)
        
        # 4. 融合空间和通道注意力
        x1 = torch.cat([sa_branch, ca_branch], dim=1)  # 通道拼接
        x1 = self.conv1x1(x1)  # 压缩并融合
        
        # 5. 低频上下文分支：池化后上采样回原尺寸
        sc_out = F.interpolate(self.SC(x), x.size()[2:], mode='nearest')  # 上采样到原尺寸
        x2 = torch.sigmoid(torch.add(x, sc_out))  # 与原始特征相加并激活
        
        # 6. 两个分支相乘融合
        out = torch.mul(x1, x2)
        
        # 7. 尾部卷积 + 残差连接
        out = self.tail(out)
        out = out + res  # 残差连接
        out = self.ReLU(out)
        return out


class NewBlock(nn.Module):
    """
    列非均匀性校正模块 (Column Non-uniformity Correction Module, CNCM)
    对应 Fig. 3 (c) 中的完整 CNCM 模块
    
    该模块内部堆叠两个 RCSSC 块，通过级联和压缩卷积实现特征融合。
    主要用于校正红外图像中的列非均匀性噪声（条纹噪声）。
    
    输入输出尺寸：保持不变 (C x H x W)
    
    结构流程（对应 Fig. 3 (c)）：
    1. 输入 -> 3x3 Conv (通道减半) -> RCSSC_1
    2. 原始输入 + RCSSC_1 输出 -> 级联 -> 3x3 Conv -> RCSSC_2
    3. RCSSC_2 输出 + 上一步级联结果 -> 级联 -> 1x1 Conv (压缩) -> 3x3 Conv
    4. 最终输出 + 原始输入 (残差连接)
    """
    def __init__(self, channel_in, reduction=16):
        super(NewBlock, self).__init__()
        # 两个 RCSSC 单元，每个处理 channel_in/2 通道
        self.unit_1 = RCSSC(int(channel_in / 2.), reduction)
        self.unit_2 = RCSSC(int(channel_in / 2.), reduction)

        # 第一个卷积：将输入通道减半，准备输入到第一个 RCSSC
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in, out_channels=int(channel_in / 2.), kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # 第二个卷积：处理级联后的特征（原始输入 + RCSSC_1 输出）
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=int(channel_in * 3 / 2.), out_channels=int(channel_in / 2.), kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        
        # 第三个卷积：压缩和融合最终特征
        # 1x1 卷积用于压缩通道，3x3 卷积用于特征融合
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=channel_in * 2, out_channels=channel_in, kernel_size=1, padding=0, stride=1),  # 压缩
            nn.Conv2d(in_channels=channel_in, out_channels=channel_in, kernel_size=3, padding=1),  # 融合
            nn.LeakyReLU()
        )

    def forward(self, x):
        residual = x  # 保存原始输入用于残差连接
        
        # 第一个 RCSSC 分支
        c1 = self.unit_1(self.conv1(x))  # 通道减半 -> RCSSC_1
        
        # 级联：原始输入 + RCSSC_1 输出
        x = torch.cat([residual, c1], 1)  # channel_in + channel_in/2 = 3*channel_in/2
        
        # 第二个 RCSSC 分支
        c2 = self.unit_2(self.conv2(x))  # 压缩到 channel_in/2 -> RCSSC_2
        
        # 级联：RCSSC_2 输出 + 上一步的级联结果
        x = torch.cat([c2, x], 1)  # channel_in/2 + 3*channel_in/2 = 2*channel_in
        
        # 压缩和融合
        x = self.conv3(x)  # 压缩回 channel_in 并融合
        
        # 残差连接
        x = torch.add(x, residual)
        return x


class RHDWTBlock(nn.Module):
    """
    残差 Haar 离散小波变换块 (Residual Haar Discrete Wavelet Transform Block)
    对应 Fig. 3 (b) 中的 RHDWT 模块（红色下采样箭头）
    
    该模块使用 Haar 小波变换进行下采样，将输入特征分解为 4 个子带：
    - 1 个低频子带 (LL): yl
    - 3 个高频子带 (LH, HL, HH): yh[0] 的三个通道
    
    实现细节（对应 Fig. 3 (b)）：
    1. 输入 Ii (C x H x W) -> HDWT 分解 -> 4 个子带，每个 (C x H/2 x W/2)
    2. 拼接 4 个子带 -> (4C x H/2 x W/2)
    3. 主干分支：3x3 Conv -> LeakyReLU -> 3x3 Conv -> I_out_model (ηC x H/2 x W/2)
    4. 残差分支：3x3 Conv (stride=2) -> I_out_res (ηC x H/2 x W/2)
    5. 输出：I_out_model + I_out_res
    
    输入： (in_channels x H x W)
    输出： (out_channels x H/2 x W/2)  # 空间尺寸减半，通道数由 out_channels 指定
    
    注意：若未安装 pytorch_wavelets，将退化为步长卷积的近似实现
    """
    def __init__(self, in_channels, out_channels, wave='haar'):
        super(RHDWTBlock, self).__init__()
        self.use_wavelet = _HAS_WAVELETS
        
        if self.use_wavelet:
            # 使用真实的小波变换
            self.dwt = DWTForward(J=1, wave=wave)  # J=1 表示一层小波分解
            # DWT 产生 1 个低频 + 3 个高频，共 4 倍通道
            # 对应 Fig. 3 (b) 中的主干分支：3x3 Conv -> LeakyReLU -> 3x3 Conv
            # 这里简化为：3x3 Conv -> LeakyReLU（第二个 3x3 Conv 在完整网络中可能在其他位置）
            self.mapper = nn.Sequential(
                nn.Conv2d(4 * in_channels, out_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(inplace=True)
            )
        else:
            # 退化实现：步长卷积模拟下采样（用于无小波库时）
            self.mapper = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(inplace=True)
            )

        # 残差分支：对应 Fig. 3 (b) 中的 I_out_res
        # 3x3 Conv (stride=2) 直接下采样
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def _transform(self, yl, yh):
        """
        将小波分解结果转换为特征图
        yl: 低频子带 (C x H/2 x W/2)
        yh: 高频子带列表，yh[0] 的形状为 (C x 3 x H/2 x W/2)，包含 3 个高频子带
        返回：拼接后的特征图 (4C x H/2 x W/2)
        """
        lst = [yl]  # 低频子带
        a = yh[0]   # 高频子带 (C x 3 x H/2 x W/2)
        # 提取 3 个高频子带
        for i in range(3):
            lst.append(a[:, :, i, :, :])  # 每个子带 (C x H/2 x W/2)
        return torch.cat(lst, 1)  # 拼接为 (4C x H/2 x W/2)

    def forward(self, x):
        if self.use_wavelet:
            # 小波分解：对应 Fig. 3 (b) 中的 HDWT 操作
            yl, yh = self.dwt(x)
            # 转换并拼接子带
            feat = self._transform(yl, yh)  # (4C x H/2 x W/2)
        else:
            # 若无小波包，直接使用输入，映射层内部负责下采样
            feat = x
        
        # 主干分支：对应 Fig. 3 (b) 中的 I_out_model
        out = self.mapper(feat)  # (out_channels x H/2 x W/2)
        
        # 残差分支：对应 Fig. 3 (b) 中的 I_out_res
        res = self.identity(x)  # (out_channels x H/2 x W/2)
        
        # 残差连接：I_out_model + I_out_res
        return out + res


class PixelShuffleUpsample(nn.Module):
    """
    像素重排上采样模块 (Pixel Shuffle Upsample)
    对应 Fig. 3 (a) 中的 Pixel Shuffle (PS) 模块（蓝色上采样箭头）
    
    该模块实现高效的上采样操作，将特征图的空间尺寸扩大 upscale_factor 倍。
    
    在 Fig. 3 (a) 的完整流程中，PS 模块包含：
    1. 1x1 Conv: 调整通道数
    2. CNCM: 列非均匀性校正（可选，在实际使用中可能需要组合 NewBlock）
    3. 3x3 Conv: 特征融合
    4. Pixel Shuffle: 上采样（本模块实现的部分）
    
    注意：本模块仅实现 Pixel Shuffle 操作，其他部分可在使用时组合。
    例如：1x1 Conv -> CNCM -> 3x3 Conv -> PixelShuffleUpsample
    
    输入要求：
    - 输入通道数必须是 upscale_factor^2 的倍数
    - 输入: (C * r^2 x H x W)，其中 r = upscale_factor
    - 输出: (C x H*r x W*r)
    
    示例：
    - 输入: (128 x 32 x 32), upscale_factor=2 -> 输出: (32 x 64 x 64)
    - 输入: (64 x 16 x 16), upscale_factor=2 -> 输出: (16 x 32 x 32)
    """
    def __init__(self, upscale_factor=2):
        super(PixelShuffleUpsample, self).__init__()
        self.ps = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        """
        执行像素重排上采样
        
        Args:
            x: 输入特征图，形状为 (B, C*r^2, H, W)
        
        Returns:
            上采样后的特征图，形状为 (B, C, H*r, W*r)
        """
        return self.ps(x)


def _fake_input(b=1, c=32, h=64, w=64):
    return torch.randn(b, c, h, w)


def demo():
    x = _fake_input()
    print("Input:", x.shape)

    # RCSSC
    rcssc = RCSSC(n_feat=32)
    y = rcssc(x)
    print("RCSSC out:", y.shape)

    # CNCM/NewBlock
    nb = NewBlock(channel_in=32)
    y2 = nb(x)
    print("NewBlock out:", y2.shape)

    # RHDWT 下采样块（输出通道提升为 64，空间降为 1/2）
    rhdwt = RHDWTBlock(in_channels=32, out_channels=64)
    y3 = rhdwt(x)
    print("RHDWTBlock out:", y3.shape)

    # PixelShuffle 上采样（需要准备 r^2 倍通道）
    ps = PixelShuffleUpsample(upscale_factor=2)
    x_ps = torch.randn(1, 32 * 4, 32, 32)
    y4 = ps(x_ps)
    print("PixelShuffleUpsample out:", y4.shape)


if __name__ == "__main__":
    demo()


