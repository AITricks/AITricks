"""
TBSN (Transformer-Based Blind-Spot Network) 即插即用模块
从网络架构中提取的核心模块，可以独立使用或集成到其他网络中

主要模块：
1. DilatedMDTA: Dilated Multi-Head Channel-wise Self-Attention (对应结构图中的 Dilated G-CSA)
2. DilatedOCA: Dilated Overlapped Cross Attention (对应结构图中的 Dilated M-WSA)
3. FeedForward: Dilated Feed-Forward Network (对应结构图中的 Dilated FFN)
4. TransformerBlock: 组合了上述模块的Transformer块
5. PatchUnshuffle/PatchShuffle: Patch操作模块
6. CentralMaskedConv2d: 中心掩码卷积（用于盲点网络）
7. OverlapPatchEmbed: Patch嵌入模块
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import numbers
from torch import einsum


# ==================== 辅助函数 ====================

def to(x):
    """获取tensor的设备和数据类型"""
    return {'device': x.device, 'dtype': x.dtype}

def to_3d(x):
    """将4D tensor转换为3D: (B, C, H, W) -> (B, H*W, C)"""
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    """将3D tensor转换为4D: (B, H*W, C) -> (B, C, H, W)"""
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

def expand_dim(t, dim, k):
    """扩展tensor的维度"""
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    """将相对位置编码转换为绝对位置编码"""
    b, l, m = x.shape
    r = (m + 1) // 2

    col_pad = torch.zeros((b, l, 1), **to(x))
    x = torch.cat((x, col_pad), dim=2)
    flat_x = rearrange(x, 'b l c -> b (l c)')
    flat_pad = torch.zeros((b, m - l), **to(x))
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=1)
    final_x = flat_x_padded.reshape(b, l + 1, m)
    final_x = final_x[:, :l, -r:]
    return final_x

def relative_logits_1d(q, rel_k):
    """计算1D相对位置logits"""
    b, h, w, _ = q.shape
    r = (rel_k.shape[0] + 1) // 2

    logits = einsum('b x y d, r d -> b x y r', q, rel_k)
    logits = rearrange(logits, 'b x y r -> (b x) y r')
    logits = rel_to_abs(logits)

    logits = logits.reshape(b, h, w, r)
    logits = expand_dim(logits, dim=2, k=r)
    return logits


# ==================== Layer Normalization ====================

class BiasFree_LayerNorm(nn.Module):
    """无偏置的LayerNorm"""
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    """带偏置的LayerNorm"""
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    """LayerNorm包装器，支持4D tensor"""
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ==================== 位置编码模块 ====================

class RelPosEmb(nn.Module):
    """相对位置编码"""
    def __init__(self, block_size, rel_size, dim_head):
        super().__init__()
        height = width = rel_size
        scale = dim_head ** -0.5

        self.block_size = block_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        block = self.block_size

        q = rearrange(q, 'b (x y) c -> b x y c', x=block)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b x i y j-> b (x y) (i j)')

        q = rearrange(q, 'b x y d -> b y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b x i y j -> b (y x) (j i)')
        return rel_logits_w + rel_logits_h


class FixedPosEmb(nn.Module):
    """固定位置编码（用于掩码注意力）"""
    def __init__(self, window_size, overlap_window_size):
        super().__init__()
        self.window_size = window_size
        self.overlap_window_size = overlap_window_size

        attention_mask_table = torch.zeros((window_size + overlap_window_size - 1), 
                                          (window_size + overlap_window_size - 1))
        attention_mask_table[0::2, :] = float('-inf')
        attention_mask_table[:, 0::2] = float('-inf')
        attention_mask_table = attention_mask_table.view(
            (window_size + overlap_window_size - 1) * (window_size + overlap_window_size - 1))

        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten_1 = torch.flatten(coords, 1)  # 2, Wh*Ww
        coords_h = torch.arange(self.overlap_window_size)
        coords_w = torch.arange(self.overlap_window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten_2 = torch.flatten(coords, 1)

        relative_coords = coords_flatten_1[:, :, None] - coords_flatten_2[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.overlap_window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.overlap_window_size - 1
        relative_coords[:, :, 0] *= self.window_size + self.overlap_window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.attention_mask = nn.Parameter(
            attention_mask_table[relative_position_index.view(-1)].view(
                1, self.window_size ** 2, self.overlap_window_size ** 2
            ), requires_grad=False)
    
    def forward(self):
        return self.attention_mask


# ==================== Patch操作模块 ====================

class PatchUnshuffle(nn.Module):
    """Patch Unshuffle操作"""
    def __init__(self, p=2, s=2):
        super().__init__()
        self.p = p
        self.s = s

    def forward(self, x):
        n, c, h, w = x.shape
        x = nn.functional.pixel_unshuffle(x, self.p)
        x = nn.functional.pixel_unshuffle(x, self.s)
        x = x.view(n, c, self.p * self.p, self.s * self.s, h//self.p//self.s, w//self.p//self.s).permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(n, c * (self.p**2) * (self.s**2), h//self.p//self.s, w//self.p//self.s)
        x = nn.functional.pixel_shuffle(x, self.p)
        return x


class PatchShuffle(nn.Module):
    """Patch Shuffle操作"""
    def __init__(self, p=2, s=2):
        super().__init__()
        self.p = p
        self.s = s

    def forward(self, x):
        n, c, h, w = x.shape
        x = nn.functional.pixel_unshuffle(x, self.p)
        x = x.view(n, c//(self.s**2), (self.s**2), (self.p**2), h//self.p, w//self.p).permute(0, 1, 3, 2, 4, 5)
        x = x.contiguous().view(n, c * (self.p**2), h//self.p, w//self.p)
        x = nn.functional.pixel_shuffle(x, self.s)
        x = nn.functional.pixel_shuffle(x, self.p)
        return x


# ==================== 卷积模块 ====================

class CentralMaskedConv2d(nn.Conv2d):
    """中心掩码卷积（用于盲点网络，中心像素不参与计算）"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kH // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


# ==================== 注意力模块 ====================

class DilatedMDTA(nn.Module):
    """
    Dilated Multi-Head Channel-wise Self-Attention
    对应结构图中的 Dilated G-CSA (Grouped Channel-wise Self-Attention)
    使用扩张卷积的通道注意力机制
    """
    def __init__(self, dim, num_heads, bias=False):
        super(DilatedMDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, 
                                   dilation=2, padding=2, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class DilatedOCA(nn.Module):
    """
    Dilated Overlapped Cross Attention
    对应结构图中的 Dilated M-WSA (Masked Window-based Self-Attention)
    使用扩张卷积的窗口注意力机制，带有掩码
    """
    def __init__(self, dim, window_size, overlap_ratio, num_heads, dim_head, bias=False):
        super(DilatedOCA, self).__init__()
        self.num_spatial_heads = num_heads
        self.dim = dim
        self.window_size = window_size
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.dim_head = dim_head
        self.inner_dim = self.dim_head * self.num_spatial_heads
        self.scale = self.dim_head**-0.5

        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size), 
                               stride=window_size, 
                               padding=(self.overlap_win_size-window_size)//2)
        self.qkv = nn.Conv2d(self.dim, self.inner_dim*3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.rel_pos_emb = RelPosEmb(
            block_size=window_size,
            rel_size=window_size + (self.overlap_win_size - window_size),
            dim_head=self.dim_head
        )
        self.fixed_pos_emb = FixedPosEmb(window_size, self.overlap_win_size)
    
    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        qs, ks, vs = qkv.chunk(3, dim=1)

        # spatial attention
        qs = rearrange(qs, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', 
                       p1=self.window_size, p2=self.window_size)
        ks, vs = map(lambda t: self.unfold(t), (ks, vs))
        ks, vs = map(lambda t: rearrange(t, 'b (c j) i -> (b i) j c', c=self.inner_dim), (ks, vs))

        # split heads
        qs, ks, vs = map(lambda t: rearrange(t, 'b n (head c) -> (b head) n c', 
                                            head=self.num_spatial_heads), (qs, ks, vs))

        # attention
        qs = qs * self.scale
        spatial_attn = (qs @ ks.transpose(-2, -1))
        spatial_attn += self.rel_pos_emb(qs)
        spatial_attn += self.fixed_pos_emb()
        spatial_attn = spatial_attn.softmax(dim=-1)

        out = (spatial_attn @ vs)

        out = rearrange(out, '(b h w head) (p1 p2) c -> b (head c) (h p1) (w p2)', 
                       head=self.num_spatial_heads, 
                       h=h // self.window_size, 
                       w=w // self.window_size, 
                       p1=self.window_size, 
                       p2=self.window_size)

        # merge spatial and channel
        out = self.project_out(out)

        return out


# ==================== Feed-Forward Network ====================

class FeedForward(nn.Module):
    """
    Dilated Feed-Forward Network
    对应结构图中的 Dilated FFN
    使用扩张卷积的前馈网络
    """
    def __init__(self, dim, ffn_expansion_factor, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=3, 
                                   stride=1, dilation=2, padding=2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=3, 
                                    stride=1, dilation=2, padding=2, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = F.gelu(x)
        x = self.project_out(x)
        return x


# ==================== Transformer Block ====================

class TransformerBlock(nn.Module):
    """
    Transformer Block (DTAB - Dilated Transformer Attention Block)
    组合了通道注意力、空间注意力和前馈网络的完整Transformer块
    对应结构图中的核心DTAB模块
    """
    def __init__(self, dim, window_size, overlap_ratio, num_channel_heads, 
                 num_spatial_heads, spatial_dim_head, ffn_expansion_factor, 
                 bias=False, LayerNorm_type='BiasFree'):
        super(TransformerBlock, self).__init__()

        self.spatial_attn = DilatedOCA(dim, window_size, overlap_ratio, 
                                     num_spatial_heads, spatial_dim_head, bias)
        self.channel_attn = DilatedMDTA(dim, num_channel_heads, bias)

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.norm3 = LayerNorm(dim, LayerNorm_type)
        self.norm4 = LayerNorm(dim, LayerNorm_type)

        self.channel_ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.spatial_ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Channel attention + FFN
        x = x + self.channel_attn(self.norm1(x))
        x = x + self.channel_ffn(self.norm2(x))
        # Spatial attention + FFN
        x = x + self.spatial_attn(self.norm3(x))
        x = x + self.spatial_ffn(self.norm4(x))
        return x


# ==================== Patch Embedding ====================

class OverlapPatchEmbed(nn.Module):
    """重叠Patch嵌入模块，使用中心掩码卷积"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = CentralMaskedConv2d(in_c, embed_dim, kernel_size=3, 
                                       stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


# ==================== 测试代码 ====================

def test_modules():
    """测试所有即插即用模块"""
    print("=" * 60)
    print("TBSN 即插即用模块测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 创建测试输入
    batch_size = 2
    channels = 48
    height, width = 64, 64
    x = torch.randn(batch_size, channels, height, width).to(device)
    
    print(f"输入形状: {x.shape}\n")
    
    # 测试1: LayerNorm
    print("1. 测试 LayerNorm...")
    layer_norm = LayerNorm(channels, 'BiasFree').to(device)
    out = layer_norm(x)
    assert out.shape == x.shape, f"LayerNorm输出形状错误: {out.shape} != {x.shape}"
    print(f"   [OK] LayerNorm: {x.shape} -> {out.shape}\n")
    
    # 测试2: CentralMaskedConv2d
    print("2. 测试 CentralMaskedConv2d...")
    masked_conv = CentralMaskedConv2d(channels, channels, kernel_size=3, padding=1).to(device)
    out = masked_conv(x)
    assert out.shape == x.shape, f"CentralMaskedConv2d输出形状错误: {out.shape} != {x.shape}"
    print(f"   [OK] CentralMaskedConv2d: {x.shape} -> {out.shape}\n")
    
    # 测试3: PatchUnshuffle
    print("3. 测试 PatchUnshuffle...")
    patch_unshuffle = PatchUnshuffle(p=2, s=2).to(device)
    out = patch_unshuffle(x)
    # PatchUnshuffle: 先unshuffle两次，然后shuffle一次p
    # 输出: (n, c * p^2 * s^2 / p^2, h//p//s * p, w//p//s * p)
    #      = (n, c * s^2, h//s, w//s)
    expected_h, expected_w = height // 2, width // 2  # h//s, w//s where s=2
    expected_c = channels * 4  # c * s^2 where s=2
    assert out.shape == (batch_size, expected_c, expected_h, expected_w), \
        f"PatchUnshuffle输出形状错误: {out.shape} != {(batch_size, expected_c, expected_h, expected_w)}"
    print(f"   [OK] PatchUnshuffle: {x.shape} -> {out.shape}\n")
    
    # 测试4: PatchShuffle (需要先unshuffle)
    print("4. 测试 PatchShuffle...")
    patch_shuffle = PatchShuffle(p=2, s=2).to(device)
    # 使用unshuffle后的输出作为输入
    unshuffled = patch_unshuffle(x)
    out = patch_shuffle(unshuffled)
    assert out.shape == x.shape, f"PatchShuffle输出形状错误: {out.shape} != {x.shape}"
    print(f"   [OK] PatchShuffle: {unshuffled.shape} -> {out.shape}\n")
    
    # 测试5: OverlapPatchEmbed
    print("5. 测试 OverlapPatchEmbed...")
    test_input = torch.randn(batch_size, 3, height, width).to(device)
    patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=channels).to(device)
    out = patch_embed(test_input)
    assert out.shape == (batch_size, channels, height, width), \
        f"OverlapPatchEmbed输出形状错误: {out.shape} != {(batch_size, channels, height, width)}"
    print(f"   [OK] OverlapPatchEmbed: {test_input.shape} -> {out.shape}\n")
    
    # 测试6: DilatedMDTA (Dilated G-CSA)
    print("6. 测试 DilatedMDTA (Dilated G-CSA)...")
    num_heads = 2
    dilated_mdta = DilatedMDTA(dim=channels, num_heads=num_heads).to(device)
    out = dilated_mdta(x)
    assert out.shape == x.shape, f"DilatedMDTA输出形状错误: {out.shape} != {x.shape}"
    print(f"   [OK] DilatedMDTA: {x.shape} -> {out.shape}\n")
    
    # 测试7: DilatedOCA (Dilated M-WSA)
    print("7. 测试 DilatedOCA (Dilated M-WSA)...")
    window_size = 8
    overlap_ratio = 0.5
    num_spatial_heads = 2
    dim_head = 16
    # 确保输入尺寸能被window_size整除
    test_h, test_w = 64, 64
    test_x = torch.randn(batch_size, channels, test_h, test_w).to(device)
    dilated_oca = DilatedOCA(dim=channels, window_size=window_size, 
                             overlap_ratio=overlap_ratio, 
                             num_heads=num_spatial_heads, 
                             dim_head=dim_head).to(device)
    out = dilated_oca(test_x)
    assert out.shape == test_x.shape, f"DilatedOCA输出形状错误: {out.shape} != {test_x.shape}"
    print(f"   [OK] DilatedOCA: {test_x.shape} -> {out.shape}\n")
    
    # 测试8: FeedForward (Dilated FFN)
    print("8. 测试 FeedForward (Dilated FFN)...")
    ffn_expansion_factor = 1
    feedforward = FeedForward(dim=channels, ffn_expansion_factor=ffn_expansion_factor).to(device)
    out = feedforward(x)
    assert out.shape == x.shape, f"FeedForward输出形状错误: {out.shape} != {x.shape}"
    print(f"   [OK] FeedForward: {x.shape} -> {out.shape}\n")
    
    # 测试9: TransformerBlock (完整DTAB)
    print("9. 测试 TransformerBlock (DTAB)...")
    transformer_block = TransformerBlock(
        dim=channels,
        window_size=window_size,
        overlap_ratio=overlap_ratio,
        num_channel_heads=num_heads,
        num_spatial_heads=num_spatial_heads,
        spatial_dim_head=dim_head,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=False,
        LayerNorm_type='BiasFree'
    ).to(device)
    out = transformer_block(test_x)
    assert out.shape == test_x.shape, f"TransformerBlock输出形状错误: {out.shape} != {test_x.shape}"
    print(f"   [OK] TransformerBlock: {test_x.shape} -> {out.shape}\n")
    
    # 测试10: 位置编码模块
    print("10. 测试位置编码模块...")
    rel_pos_emb = RelPosEmb(block_size=window_size, rel_size=window_size*2, dim_head=dim_head).to(device)
    # 创建测试query: (b, block_size*block_size, dim_head)
    # RelPosEmb会将q reshape为 (b, block_size, block_size, dim_head)
    test_q = torch.randn(batch_size, window_size*window_size, dim_head).to(device)
    rel_out = rel_pos_emb(test_q)
    print(f"   [OK] RelPosEmb: query {test_q.shape} -> {rel_out.shape}")
    
    fixed_pos_emb = FixedPosEmb(window_size=window_size, overlap_window_size=window_size*2).to(device)
    fixed_mask = fixed_pos_emb()
    print(f"   [OK] FixedPosEmb: mask shape {fixed_mask.shape}\n")
    
    # 测试11: 完整前向传播流程
    print("11. 测试完整前向传播流程...")
    # 模拟一个简单的网络流程
    test_img = torch.randn(batch_size, 3, 64, 64).to(device)
    
    # Patch Embedding
    patch_embed = OverlapPatchEmbed(in_c=3, embed_dim=channels).to(device)
    embedded = patch_embed(test_img)
    
    # Transformer Block
    transformer = TransformerBlock(
        dim=channels,
        window_size=8,
        overlap_ratio=0.5,
        num_channel_heads=2,
        num_spatial_heads=2,
        spatial_dim_head=16,
        ffn_expansion_factor=1,
        bias=False,
        LayerNorm_type='BiasFree'
    ).to(device)
    transformed = transformer(embedded)
    
    # 输出层
    output_conv = nn.Conv2d(channels, 3, kernel_size=1).to(device)
    output = output_conv(transformed)
    
    assert output.shape == test_img.shape, f"完整流程输出形状错误: {output.shape} != {test_img.shape}"
    print(f"   [OK] 完整流程: {test_img.shape} -> {output.shape}\n")
    
    print("=" * 60)
    print("所有测试通过！[OK]")
    print("=" * 60)


if __name__ == '__main__':
    test_modules()