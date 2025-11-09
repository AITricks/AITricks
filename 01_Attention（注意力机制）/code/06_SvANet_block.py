"""
测试 SvANet 核心即插即用模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Callable, Any, Union, Sequence, Tuple


# ==================== 基础工具类和函数 ====================

def pair(val):
    """转换为元组"""
    return val if isinstance(val, (tuple, list)) else (val, val)


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """确保值能被 divisor 整除"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def shuffle_tensor(feature: torch.Tensor, mode: int = 1) -> list:
    """打乱张量"""
    if isinstance(feature, torch.Tensor):
        feature = [feature]
    indexs = None
    output = []
    for f in feature:
        B, C, H, W = f.shape
        if mode == 1:
            f = f.flatten(2)
            if indexs is None:
                indexs = torch.randperm(f.shape[-1], device=f.device)
            f = f[:, :, indexs.to(f.device)]
            f = f.reshape(B, C, H, W)
        output.append(f)
    return output


def set_method(self, element_name, element_value):
    """设置属性"""
    return setattr(self, element_name, element_value)


def call_method(self, element_name):
    """获取属性"""
    return getattr(self, element_name)


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    """自适应平均池化"""
    def __init__(self, output_size: int or tuple = 1):
        super(AdaptiveAvgPool2d, self).__init__(output_size=output_size)


class BaseConv2d(nn.Module):
    """基础卷积层"""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: Optional[bool] = None,
        BNorm: bool = False,
        ActLayer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any
    ):
        super(BaseConv2d, self).__init__()
        if padding is None:
            padding = int((kernel_size - 1) // 2 * dilation)
        if bias is None:
            bias = not BNorm
        
        self.Conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        )
        self.Bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.1) if BNorm else nn.Identity()
        
        if ActLayer is not None:
            self.Act = ActLayer(inplace=True) if ActLayer != nn.Sigmoid else ActLayer()
        else:
            self.Act = None
    
    def forward(self, x):
        x = self.Conv(x)
        x = self.Bn(x)
        if self.Act is not None:
            x = self.Act(x)
        return x


class StochasticDepth(nn.Module):
    """随机深度"""
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        if self.training and torch.rand(1) < self.p:
            return x * 0.0
        return x


class Dropout(nn.Module):
    """Dropout层"""
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout(p) if p > 0 else nn.Identity()
    
    def forward(self, x):
        return self.dropout(x)


# ==================== 核心即插即用模块 ====================

class MoCAttention(nn.Module):
    """Monte Carlo 注意力 - 学习全局和局部特征"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        SqueezeFactor: int = 4,
        PoolRes: list = [1, 2, 3],
        Act: Callable[..., nn.Module] = nn.ReLU,
        ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
        MoCOrder: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = max(make_divisible(InChannels // SqueezeFactor, 8), 32)
        
        AllPoolRes = PoolRes + [1] if 1 not in PoolRes else PoolRes
        for k in AllPoolRes:
            Pooling = AdaptiveAvgPool2d(k)
            set_method(self, 'Pool%d' % k, Pooling)
        
        self.SELayer = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
        
        self.PoolRes = PoolRes
        self.MoCOrder = MoCOrder
    
    def monte_carlo_sample(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            PoolKeep = np.random.choice(self.PoolRes)
            x1 = shuffle_tensor(x)[0] if self.MoCOrder else x
            AttnMap: torch.Tensor = call_method(self, 'Pool%d' % PoolKeep)(x1)
            if AttnMap.shape[-1] > 1:
                AttnMap = AttnMap.flatten(2)
                AttnMap = AttnMap[:, :, torch.randperm(AttnMap.shape[-1])[0]]
                AttnMap = AttnMap[:, :, None, None]
        else:
            AttnMap: torch.Tensor = call_method(self, 'Pool%d' % 1)(x)
        
        return AttnMap
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        AttnMap = self.monte_carlo_sample(x)
        return x * self.SELayer(AttnMap)


class SqueezeExcitation(nn.Module):
    """SE 注意力机制"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        SqueezeFactor: int = 4,
        Act: Callable[..., nn.Module] = nn.ReLU,
        ScaleAct: Callable[..., nn.Module] = nn.Sigmoid,
        **kwargs: Any,
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = max(make_divisible(InChannels // SqueezeFactor, 8), 32)
        
        self.SELayer = nn.Sequential(
            AdaptiveAvgPool2d(1),
            BaseConv2d(InChannels, HidChannels, 1, ActLayer=Act),
            BaseConv2d(HidChannels, InChannels, 1, ActLayer=ScaleAct),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.SELayer(x)


class LinearSelfAttention(nn.Module):
    """线性复杂度自注意力机制 (MobileViTv2风格)"""
    
    def __init__(
        self,
        DimEmbed: int,
        AttnDropRate: float = 0.0,
        Bias: bool = True,
    ):
        super().__init__()
        self.qkv_proj = BaseConv2d(DimEmbed, 1 + (2 * DimEmbed), 1, bias=Bias)
        self.AttnDropRate = Dropout(p=AttnDropRate)
        self.out_proj = BaseConv2d(DimEmbed, DimEmbed, 1, bias=Bias)
        self.DimEmbed = DimEmbed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, P, N] -> [B, 1+2C, P, N]
        qkv = self.qkv_proj(x)
        
        # 分割为 query, key, value
        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.DimEmbed, self.DimEmbed], dim=1
        )
        
        # 沿N维度应用softmax
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.AttnDropRate(context_scores)
        
        # 计算上下文向量
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        
        # 结合上下文向量和values
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Module):
    """线性注意力FFN模块"""
    
    def __init__(
        self,
        DimEmbed: int,
        DimFfnLatent: int,
        AttnDropRate: float = 0.0,
        DropRate: float = 0.1,
        FfnDropRate: float = 0.0,
    ):
        super().__init__()
        AttnUnit = LinearSelfAttention(DimEmbed, AttnDropRate, Bias=True)
        
        self.PreNormAttn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            AttnUnit,
            Dropout(DropRate),
        )
        
        self.PreNormFfn = nn.Sequential(
            nn.BatchNorm2d(DimEmbed),
            BaseConv2d(DimEmbed, DimFfnLatent, 1, 1, ActLayer=nn.SiLU),
            Dropout(FfnDropRate),
            BaseConv2d(DimFfnLatent, DimEmbed, 1, 1),
            Dropout(DropRate),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        x = x + self.PreNormAttn(x)
        # 前馈网络
        x = x + self.PreNormFfn(x)
        return x


class AssembleFormer(nn.Module):
    """组合 CNN 和 ViT 的混合模块 - 完整实现"""
    
    def __init__(
        self,
        InChannels: int,
        FfnMultiplier: Union[float, Sequence[float]] = 2.0,
        NumAttnBlocks: int = 2,
        PatchRes: int = 2,
        Dilation: int = 1,
        AttnDropRate: float = 0.0,
        DropRate: float = 0.0,
        FfnDropRate: float = 0.0,
        SDProb: float = 0.0,
        ViTSELayer: Optional[nn.Module] = None,
        **kwargs: Any,
    ):
        super().__init__()
        DimAttnUnit = InChannels // 2
        
        # 局部表示分支
        Conv3x3In = BaseConv2d(
            InChannels, InChannels, 3, 1, dilation=Dilation, 
            BNorm=True, ActLayer=nn.SiLU,
        )
        ViTSELayer = ViTSELayer(InChannels, **kwargs) if ViTSELayer is not None else nn.Identity()
        Conv1x1In = BaseConv2d(InChannels, DimAttnUnit, 1, 1, bias=False)
        self.LocalRep = nn.Sequential(Conv3x3In, ViTSELayer, Conv1x1In)
        
        # 全局表示分支 (Transformer)
        self.GlobalRep = self._build_attn_layer(
            DimAttnUnit, FfnMultiplier, NumAttnBlocks, AttnDropRate, DropRate, FfnDropRate
        )
        
        # 投影层: 局部 + 全局 -> 输出
        self.ConvProj = BaseConv2d(2 * DimAttnUnit, InChannels, 1, 1, BNorm=True)
        
        # Patch相关参数
        self.HPatch, self.WPatch = pair(PatchRes)
        self.PatchArea = self.WPatch * self.HPatch
        
        # 随机深度
        self.Dropout = StochasticDepth(SDProb)
    
    def _build_attn_layer(
        self,
        DimModel: int,
        FfnMult: Union[Sequence, float],
        NumAttnBlocks: int,
        AttnDropRate: float,
        DropRate: float,
        FfnDropRate: float,
    ) -> nn.Module:
        """构建注意力层"""
        if isinstance(FfnMult, Sequence) and len(FfnMult) == 2:
            DimFfn = np.linspace(FfnMult[0], FfnMult[1], NumAttnBlocks, dtype=float) * DimModel
        elif isinstance(FfnMult, Sequence) and len(FfnMult) == 1:
            DimFfn = [FfnMult[0] * DimModel] * NumAttnBlocks
        elif isinstance(FfnMult, (int, float)):
            DimFfn = [FfnMult * DimModel] * NumAttnBlocks
        else:
            raise NotImplementedError
        
        # 确保维度是16的倍数
        DimFfn = [make_divisible(d, 16) for d in DimFfn]
        
        GlobalRep = [
            LinearAttnFFN(DimModel, DimFfn[block_idx], AttnDropRate, DropRate, FfnDropRate)
            for block_idx in range(NumAttnBlocks)
        ]
        GlobalRep.append(nn.BatchNorm2d(DimModel))
        return nn.Sequential(*GlobalRep)
    
    def unfolding(self, FeatureMap: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """将特征图展开为patches"""
        B, C, H, W = FeatureMap.shape
        # [B, C, H, W] -> [B, C*P, N] -> [B, C, P, N]
        Patches = F.unfold(
            FeatureMap,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )
        Patches = Patches.reshape(B, C, self.HPatch * self.WPatch, -1)
        return Patches, (H, W)
    
    def folding(self, Patches: torch.Tensor, OutputSize: Tuple[int, int]) -> torch.Tensor:
        """将patches折叠回特征图"""
        B, C, P, N = Patches.shape
        # [B, C, P, N] -> [B, C*P, N]
        Patches = Patches.reshape(B, C * P, N)
        FeatureMap = F.fold(
            Patches,
            output_size=OutputSize,
            kernel_size=(self.HPatch, self.WPatch),
            stride=(self.HPatch, self.WPatch),
        )
        return FeatureMap
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, C//2, H, W] 局部特征
        FmConv = self.LocalRep(x)
        
        # 转换为patches: [B, C//2, H, W] -> [B, C//2, P, N]
        Patches, OutputSize = self.unfolding(FmConv)
        
        # Transformer处理: [B, C//2, P, N] -> [B, C//2, P, N]
        Patches = self.GlobalRep(Patches)
        
        # 折叠回特征图: [B, C//2, P, N] -> [B, C//2, H, W]
        Fm = self.folding(Patches, OutputSize)
        
        # 局部 + 全局: [B, C//2, H, W] + [B, C//2, H, W] -> [B, C, H, W]
        Fm = self.ConvProj(torch.cat((Fm, FmConv), dim=1))
        
        # 残差连接
        return x + self.Dropout(Fm)


class FGBottleneck(nn.Module):
    """特征引导瓶颈块 (对应结构图中的 MCBottleneck)"""
    
    def __init__(
        self,
        InChannels: int,
        HidChannels: int = None,
        Expansion: float = 2.0,
        Stride: int = 1,
        Dilation: int = 1,
        DropRate: float = 0.0,
        SELayer: Optional[nn.Module] = None,
        ActLayer: Optional[Callable[..., nn.Module]] = None,
        ViTBlock: Optional[nn.Module] = None,
        **kwargs: Any
    ):
        super().__init__()
        if HidChannels is None:
            HidChannels = make_divisible(InChannels * Expansion, 8)
        
        self.Bottleneck = nn.Sequential(
            BaseConv2d(InChannels, HidChannels, 1, BNorm=True, ActLayer=nn.ReLU),
            BaseConv2d(HidChannels, HidChannels, 3, Stride, dilation=Dilation, BNorm=True, ActLayer=nn.ReLU),
            SELayer(InChannels=HidChannels, **kwargs) if SELayer is not None else nn.Identity(),
            BaseConv2d(HidChannels, InChannels, 1, BNorm=True)
        )
        
        self.ActLayer = ActLayer(inplace=True) if ActLayer is not None else nn.Identity()
        self.Dropout = StochasticDepth(DropRate) if DropRate > 0 else nn.Identity()
        self.ViTLayer = ViTBlock(InChannels, **kwargs) if ViTBlock is not None else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Out = self.Bottleneck(x)
        Out = self.ActLayer(x + self.Dropout(Out))
        return self.ViTLayer(Out)


# ==================== 测试函数 ====================

def test_modules():
    """测试所有核心模块"""
    print("=" * 80)
    print("开始测试 SvANet 核心即插即用模块")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建测试输入
    batch_size = 2
    in_channels = 64
    h, w = 32, 32
    x = torch.randn(batch_size, in_channels, h, w).to(device)
    
    print(f"输入形状: {x.shape}\n")
    
    # 测试 1: MoCAttention
    print("测试 1: MoCAttention (Monte Carlo 注意力)")
    moc_attn = MoCAttention(InChannels=in_channels).to(device)
    out = moc_attn(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] MoCAttention 测试通过\n")
    
    # 测试 2: SqueezeExcitation
    print("测试 2: SqueezeExcitation (SE 注意力)")
    se = SqueezeExcitation(InChannels=in_channels).to(device)
    out = se(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] SqueezeExcitation 测试通过\n")
    
    # 测试 3: AssembleFormer
    print("测试 3: AssembleFormer (CNN + ViT 混合模块)")
    assem_former = AssembleFormer(InChannels=in_channels, NumAttnBlocks=1).to(device)
    out = assem_former(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] AssembleFormer 测试通过\n")
    
    # 测试 4: FGBottleneck (MCBottleneck)
    print("测试 4: FGBottleneck (特征引导瓶颈)")
    fg_bottleneck = FGBottleneck(InChannels=in_channels, SELayer=None).to(device)
    out = fg_bottleneck(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] FGBottleneck 测试通过\n")
    
    # 测试 5: 使用 MoCAttention 的 FGBottleneck
    print("测试 5: FGBottleneck + MoCAttention")
    fg_bottleneck_se = FGBottleneck(
        InChannels=in_channels, 
        SELayer=MoCAttention
    ).to(device)
    out = fg_bottleneck_se(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] FGBottleneck + MoCAttention 测试通过\n")
    
    # 测试 6: FGBottleneck + MoCAttention + AssembleFormer
    print("测试 6: FGBottleneck + MoCAttention + AssembleFormer (完整MCBottleneck)")
    fg_bottleneck_full = FGBottleneck(
        InChannels=in_channels,
        SELayer=MoCAttention,
        ViTBlock=AssembleFormer,
        NumAttnBlocks=1,
    ).to(device)
    out = fg_bottleneck_full(x)
    print(f"  输入: {x.shape}")
    print(f"  输出: {out.shape}")
    print(f"  [OK] FGBottleneck + MoCAttention + AssembleFormer 测试通过\n")
    
    print("=" * 80)
    print("所有核心即插即用模块测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    test_modules()

