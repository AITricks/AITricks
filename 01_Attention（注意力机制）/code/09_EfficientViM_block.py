"""
EfficientViM 即插即用模块
包含 HSM-SSD 层和 EfficientViMBlock，可直接替换到其他 Vision Transformer 模型中

HSM-SSD: Hidden State Memory - State Space Discretization layer
优化了计算复杂度，将 O(LD^2) 操作替换为 O(ND^2) 操作
"""

import math
import torch
import torch.nn as nn


# ==================== 工具类 ====================

class LayerNorm2D(nn.Module):
    """LayerNorm for channels of 2D tensor(B C H W)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm2D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, H, W)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, H, W)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class LayerNorm1D(nn.Module):
    """LayerNorm for channels of 1D tensor(B C L)"""
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super(LayerNorm1D, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_channels, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, L)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, L)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)  # (B, C, L)

        if self.affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class ConvLayer2D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm2d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer2D, self).__init__()
        self.conv = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class ConvLayer1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, norm=nn.BatchNorm1d, act_layer=nn.ReLU, bn_weight_init=1):
        super(ConvLayer1D, self).__init__()
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        self.norm = norm(num_features=out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        
        if self.norm:
            torch.nn.init.constant_(self.norm.weight, bn_weight_init)
            torch.nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class FFN(nn.Module):
    """Feed-Forward Network"""
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2D(in_dim, dim, 1)
        self.fc2 = ConvLayer2D(dim, in_dim, 1, act_layer=None, bn_weight_init=0)
        
    def forward(self, x):
        x = self.fc2(self.fc1(x))
        return x


# ==================== 核心即插即用模块 ====================

class HSMSSD(nn.Module):
    """
    Hidden State Memory - State Space Discretization (HSM-SSD) 层
    
    这是一个即插即用的注意力模块，可以替换 Vision Transformer 中的注意力层。
    优化了计算复杂度：将 O(LD^2) 操作替换为 O(ND^2) 操作，其中 N 是隐藏状态维度。
    
    Args:
        d_model: 模型维度
        ssd_expand: SSD 扩展因子，默认 1
        A_init_range: A 参数的初始化范围，默认 (1, 16)
        state_dim: 隐藏状态维度，默认 64
    
    输入: (B, C, H, W) 或 (B, C, L) where L = H*W
    输出: (B, C, H, W), (B, C, N) - 输出特征和隐藏状态
    """
    def __init__(self, d_model, ssd_expand=1, A_init_range=(1, 16), state_dim=64):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * d_model)
        self.state_dim = state_dim

        # BCdt 投影层 (3*state_dim)
        self.BCdt_proj = ConvLayer1D(d_model, 3*state_dim, 1, norm=None, act_layer=None)
        
        # 2D 深度卷积处理空间信息
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2D(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None, act_layer=None, bn_weight_init=0)
        
        # HSM 组件中的投影层 (将 O(LD^2) 优化为 O(ND^2))
        self.hz_proj = ConvLayer1D(d_model, 2*self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1D(self.d_inner, d_model, 1, norm=None, act_layer=None, bn_weight_init=0)

        # 状态空间参数初始化
        A = torch.empty(self.state_dim, dtype=torch.float32).uniform_(*A_init_range)
        self.A = torch.nn.Parameter(A)
        self.act = nn.SiLU()
        
        # 跳跃连接权重
        self.D = nn.Parameter(torch.ones(1))
        self.D._no_weight_decay = True

    def forward(self, x):
        """
        Args:
            x: (B, C, L) where L = H*W
        
        Returns:
            y: (B, C, H, W) - 输出特征
            h: (B, C, N) - 隐藏状态
        """
        batch, _, L = x.shape
        H = int(math.sqrt(L))
        
        # 离散化: 生成 A, B, C 参数
        BCdt = self.dw(self.BCdt_proj(x).view(batch, -1, H, H)).flatten(2)
        B, C, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        A = (dt + self.A.view(1, -1, 1)).softmax(-1)
        
        # 状态空间计算: h = x @ (A * B)^T
        AB = (A * B)
        h = x @ AB.transpose(-2, -1)  # (B, C, N)
        
        # HSM 组件: 在隐藏状态空间中进行计算 (O(ND^2) 复杂度)
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        
        # 输出: y = h @ C
        y = h @ C  # (B, C, N) @ (B, C, L) -> (B, C, L)
        
        y = y.view(batch, -1, H, H).contiguous()  # (B, C, H, W)
        return y, h


class EfficientViMBlock(nn.Module):
    """
    EfficientViM Block - 完整的即插即用模块
    
    包含:
    - 3x3 深度卷积 (DWConv)
    - HSM-SSD 注意力层
    - 另一个 3x3 深度卷积
    - 前馈网络 (FFN)
    
    每个组件都有可学习的 LayerScale 权重，用于稳定的训练。
    
    Args:
        dim: 特征维度
        mlp_ratio: MLP 扩展比例，默认 4.0
        ssd_expand: SSD 扩展因子，默认 1
        state_dim: 隐藏状态维度，默认 64
    
    输入: (B, C, H, W)
    输出: (B, C, H, W), (B, C, N) - 输出特征和隐藏状态
    """
    def __init__(self, dim, mlp_ratio=4., ssd_expand=1, state_dim=64):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        # HSM-SSD 注意力层
        self.mixer = HSMSSD(d_model=dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm1D(dim)
        
        # 深度卷积层
        self.dwconv1 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        self.dwconv2 = ConvLayer2D(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None)
        
        # 前馈网络
        self.ffn = FFN(in_dim=dim, dim=int(dim * mlp_ratio))
        
        # LayerScale: 可学习的残差连接权重
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim), requires_grad=True)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            x: (B, C, H, W) - 输出特征
            h: (B, C, N) - 隐藏状态 (来自 HSM-SSD)
        """
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)
        
        # DWConv1
        x = (1 - alpha[0]) * x + alpha[0] * self.dwconv1(x)
        
        # HSM-SSD
        x_prev = x
        x, h = self.mixer(self.norm(x.flatten(2)))  # (B, C, L) -> (B, C, H, W), (B, C, N)
        x = (1 - alpha[1]) * x_prev + alpha[1] * x
        
        # DWConv2
        x = (1 - alpha[2]) * x + alpha[2] * self.dwconv2(x)
        
        # FFN
        x = (1 - alpha[3]) * x + alpha[3] * self.ffn(x)
        
        return x, h


# ==================== 测试代码 ====================

def test_hsm_ssd():
    """测试 HSMSSD 模块"""
    print("=" * 50)
    print("测试 HSMSSD 模块")
    print("=" * 50)
    
    # 创建模块
    d_model = 128
    state_dim = 64
    hsm_ssd = HSMSSD(d_model=d_model, state_dim=state_dim)
    
    # 创建测试输入 (B, C, H, W) -> flatten 为 (B, C, L)
    batch_size = 2
    H, W = 14, 14  # 典型的 patch 尺寸
    L = H * W
    
    # 输入格式: (B, C, L)
    x = torch.randn(batch_size, d_model, L)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    y, h = hsm_ssd(x)
    
    print(f"输出 y 形状: {y.shape}")
    print(f"隐藏状态 h 形状: {h.shape}")
    print(f"✓ HSMSSD 测试通过")
    print()


def test_efficient_vim_block():
    """测试 EfficientViMBlock 模块"""
    print("=" * 50)
    print("测试 EfficientViMBlock 模块")
    print("=" * 50)
    
    # 创建模块
    dim = 128
    block = EfficientViMBlock(dim=dim, mlp_ratio=4.0, state_dim=64)
    
    # 创建测试输入 (B, C, H, W)
    batch_size = 2
    H, W = 14, 14
    
    x = torch.randn(batch_size, dim, H, W)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    x_out, h = block(x)
    
    print(f"输出 x_out 形状: {x_out.shape}")
    print(f"隐藏状态 h 形状: {h.shape}")
    print(f"✓ EfficientViMBlock 测试通过")
    print()


def test_usage_example():
    """展示如何在其他模型中使用即插即用模块"""
    print("=" * 50)
    print("即插即用使用示例")
    print("=" * 50)
    
    # 模拟一个简单的 Vision Transformer block
    class SimpleViTBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            # 使用 EfficientViMBlock 替换原来的注意力层
            self.block = EfficientViMBlock(dim=dim, state_dim=64)
            
        def forward(self, x):
            x, h = self.block(x)
            return x
    
    # 测试
    model = SimpleViTBlock(dim=256)
    x = torch.randn(1, 256, 14, 14)
    out = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}")
    print(f"✓ 即插即用示例测试通过")
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("EfficientViM 即插即用模块测试")
    print("=" * 60 + "\n")
    
    try:
        # 测试各个模块
        test_hsm_ssd()
        test_efficient_vim_block()
        test_usage_example()
        
        print("=" * 60)
        print("所有测试通过！✓")
        print("=" * 60)
        
        print("\n使用说明:")
        print("1. HSMSSD: 核心注意力模块，输入 (B, C, L)，输出 (B, C, H, W) 和 (B, C, N)")
        print("2. EfficientViMBlock: 完整块，输入 (B, C, H, W)，输出 (B, C, H, W) 和 (B, C, N)")
        print("3. 可以直接替换 ViT、Swin 等模型中的注意力层或 block")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

