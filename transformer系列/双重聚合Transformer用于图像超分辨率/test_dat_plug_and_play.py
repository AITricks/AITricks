"""
DAT 即插即用模块测试
包含完整模块代码和简单测试
"""

import torch
import torch.nn as nn
from einops import rearrange


# ==================== 辅助函数 ====================

def img2windows(img, H_sp, W_sp):
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp* W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


# ==================== 模块代码 ====================

class SpatialGate(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x.shape
        x2 = self.norm(x2)
        x2 = x2.transpose(1, 2).contiguous()
        x2 = x2.view(B, C//2, H, W)
        x2 = self.conv(x2)
        x2 = x2.flatten(2)
        x2 = x2.transpose(-1, -2).contiguous()
        return x1 * x2


class SGFN(nn.Module):
    """空间门控前馈网络"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.sg(x, H, W)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads, residual):
        super().__init__()
        self.residual = residual
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True), nn.Linear(self.pos_dim, self.num_heads))
    
    def forward(self, biases):
        if self.residual:
            pos = self.pos_proj(biases)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos


class Spatial_Attention(nn.Module):
    def __init__(self, dim, idx, split_size=[8,8], dim_out=None, num_heads=6, attn_drop=0., proj_drop=0., qk_scale=None, position_bias=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.split_size = split_size
        self.num_heads = num_heads
        self.idx = idx
        self.position_bias = position_bias
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if idx == 0:
            H_sp, W_sp = self.split_size[0], self.split_size[1]
        elif idx == 1:
            W_sp, H_sp = self.split_size[0], self.split_size[1]
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads, residual=False)
            position_bias_h = torch.arange(1 - self.H_sp, self.H_sp)
            position_bias_w = torch.arange(1 - self.W_sp, self.W_sp)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w], indexing='ij'))
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()
            self.register_buffer('rpe_biases', biases)
            coords_h = torch.arange(self.H_sp)
            coords_w = torch.arange(self.W_sp)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.H_sp - 1
            relative_coords[:, :, 1] += self.W_sp - 1
            relative_coords[:, :, 0] *= 2 * self.W_sp - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index', relative_position_index)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2win(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(-2,-1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp* self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def forward(self, qkv, H, W, mask=None):
        q,k,v = qkv[0], qkv[1], qkv[2]
        B, L, C = q.shape
        assert L == H * W
        q = self.im2win(q, H, W)
        k = self.im2win(k, H, W)
        v = self.im2win(v, H, W)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.position_bias:
            pos = self.pos(self.rpe_biases)
            relative_position_bias = pos[self.relative_position_index.view(-1)].view(
                self.H_sp * self.W_sp, self.H_sp * self.W_sp, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)
        N = attn.shape[3]
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp* self.W_sp, C)
        x = windows2img(x, self.H_sp, self.W_sp, H, W)
        return x


class Adaptive_Spatial_Attention(nn.Module):
    """自适应空间注意力"""
    def __init__(self, dim, num_heads, reso=64, split_size=[8,8], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.shift_size = shift_size
        self.b_idx = b_idx
        self.rg_idx = rg_idx
        self.patches_resolution = reso
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        assert 0 <= self.shift_size[0] < self.split_size[0]
        assert 0 <= self.shift_size[1] < self.split_size[1]
        self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.attns = nn.ModuleList([
            Spatial_Attention(dim//2, idx=i, split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                             qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, position_bias=True)
            for i in range(self.branch_num)])
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
            attn_mask = self.calculate_mask(self.patches_resolution, self.patches_resolution)
            self.register_buffer("attn_mask_0", attn_mask[0])
            self.register_buffer("attn_mask_1", attn_mask[1])
        else:
            attn_mask = None
            self.register_buffer("attn_mask_0", None)
            self.register_buffer("attn_mask_1", None)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def calculate_mask(self, H, W):
        img_mask_0 = torch.zeros((1, H, W, 1))
        img_mask_1 = torch.zeros((1, H, W, 1))
        h_slices_0 = (slice(0, -self.split_size[0]), slice(-self.split_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        w_slices_0 = (slice(0, -self.split_size[1]), slice(-self.split_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        h_slices_1 = (slice(0, -self.split_size[1]), slice(-self.split_size[1], -self.shift_size[1]), slice(-self.shift_size[1], None))
        w_slices_1 = (slice(0, -self.split_size[0]), slice(-self.split_size[0], -self.shift_size[0]), slice(-self.shift_size[0], None))
        cnt = 0
        for h in h_slices_0:
            for w in w_slices_0:
                img_mask_0[:, h, w, :] = cnt
                cnt += 1
        cnt = 0
        for h in h_slices_1:
            for w in w_slices_1:
                img_mask_1[:, h, w, :] = cnt
                cnt += 1
        img_mask_0 = img_mask_0.view(1, H // self.split_size[0], self.split_size[0], W // self.split_size[1], self.split_size[1], 1)
        img_mask_0 = img_mask_0.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[0], self.split_size[1], 1)
        mask_windows_0 = img_mask_0.view(-1, self.split_size[0] * self.split_size[1])
        attn_mask_0 = mask_windows_0.unsqueeze(1) - mask_windows_0.unsqueeze(2)
        attn_mask_0 = attn_mask_0.masked_fill(attn_mask_0 != 0, float(-100.0)).masked_fill(attn_mask_0 == 0, float(0.0))
        img_mask_1 = img_mask_1.view(1, H // self.split_size[1], self.split_size[1], W // self.split_size[0], self.split_size[0], 1)
        img_mask_1 = img_mask_1.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.split_size[1], self.split_size[0], 1)
        mask_windows_1 = img_mask_1.view(-1, self.split_size[1] * self.split_size[0])
        attn_mask_1 = mask_windows_1.unsqueeze(1) - mask_windows_1.unsqueeze(2)
        attn_mask_1 = attn_mask_1.masked_fill(attn_mask_1 != 0, float(-100.0)).masked_fill(attn_mask_1 == 0, float(0.0))
        return attn_mask_0, attn_mask_1

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W
        qkv = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3)
        v = qkv[2].transpose(-2,-1).contiguous().view(B, C, H, W)
        max_split_size = max(self.split_size[0], self.split_size[1])
        pad_l = pad_t = 0
        pad_r = (max_split_size - W % max_split_size) % max_split_size
        pad_b = (max_split_size - H % max_split_size) % max_split_size
        qkv = qkv.reshape(3*B, H, W, C).permute(0, 3, 1, 2)
        qkv = torch.nn.functional.pad(qkv, (pad_l, pad_r, pad_t, pad_b)).reshape(3, B, C, -1).transpose(-2, -1)
        _H = pad_b + H
        _W = pad_r + W
        _L = _H * _W
        if (self.rg_idx % 2 == 0 and self.b_idx > 0 and (self.b_idx - 2) % 4 == 0) or (self.rg_idx % 2 != 0 and self.b_idx % 4 == 0):
            qkv = qkv.view(3, B, _H, _W, C)
            qkv_0 = torch.roll(qkv[:,:,:,:,:C//2], shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
            qkv_0 = qkv_0.view(3, B, _L, C//2)
            qkv_1 = torch.roll(qkv[:,:,:,:,C//2:], shifts=(-self.shift_size[1], -self.shift_size[0]), dims=(2, 3))
            qkv_1 = qkv_1.view(3, B, _L, C//2)
            if self.patches_resolution != _H or self.patches_resolution != _W:
                mask_tmp = self.calculate_mask(_H, _W)
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=mask_tmp[0].to(x.device))
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=mask_tmp[1].to(x.device))
            else:
                x1_shift = self.attns[0](qkv_0, _H, _W, mask=self.attn_mask_0)
                x2_shift = self.attns[1](qkv_1, _H, _W, mask=self.attn_mask_1)
            x1 = torch.roll(x1_shift, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x2 = torch.roll(x2_shift, shifts=(self.shift_size[1], self.shift_size[0]), dims=(1, 2))
            x1 = x1[:, :H, :W, :].reshape(B, L, C//2)
            x2 = x2[:, :H, :W, :].reshape(B, L, C//2)
            attened_x = torch.cat([x1,x2], dim=2)
        else:
            x1 = self.attns[0](qkv[:,:,:,:C//2], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            x2 = self.attns[1](qkv[:,:,:,C//2:], _H, _W)[:, :H, :W, :].reshape(B, L, C//2)
            attened_x = torch.cat([x1,x2], dim=2)
        conv_x = self.dwconv(v)
        channel_map = self.channel_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, 1, C)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        spatial_map = self.spatial_interaction(attention_reshape)
        attened_x = attened_x * torch.sigmoid(channel_map)
        conv_x = torch.sigmoid(spatial_map) * conv_x
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Adaptive_Channel_Attention(nn.Module):
    """自适应通道注意力"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.GELU(),
            nn.Conv2d(dim // 8, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 16, kernel_size=1),
            nn.BatchNorm2d(dim // 16),
            nn.GELU(),
            nn.Conv2d(dim // 16, 1, kernel_size=1)
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        v_ = v.reshape(B, C, N).contiguous().view(B, C, H, W)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attened_x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        conv_x = self.dwconv(v_)
        attention_reshape = attened_x.transpose(-2,-1).contiguous().view(B, C, H, W)
        channel_map = self.channel_interaction(attention_reshape)
        spatial_map = self.spatial_interaction(conv_x).permute(0, 2, 3, 1).contiguous().view(B, N, 1)
        attened_x = attened_x * torch.sigmoid(spatial_map)
        conv_x = conv_x * torch.sigmoid(channel_map)
        conv_x = conv_x.permute(0, 2, 3, 1).contiguous().view(B, N, C)
        x = attened_x + conv_x
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ==================== 包装类：用于图像格式输入 ====================

class SGFN_Wrapper(nn.Module):
    """SGFN包装类，支持图像格式输入输出"""
    def __init__(self, channel, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or channel * 4
        self.sgfn = SGFN(in_features=channel, hidden_features=hidden_features, out_features=channel, act_layer=act_layer, drop=drop)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        out_seq = self.sgfn(x_seq, H, W)
        out = rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)
        return out


class Adaptive_Channel_Attention_Wrapper(nn.Module):
    """Adaptive_Channel_Attention包装类，支持图像格式输入输出"""
    def __init__(self, channel, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.attn = Adaptive_Channel_Attention(dim=channel, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        out_seq = self.attn(x_seq, H, W)
        out = rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)
        return out


class Adaptive_Spatial_Attention_Wrapper(nn.Module):
    """Adaptive_Spatial_Attention包装类，支持图像格式输入输出"""
    def __init__(self, channel, num_heads=8, reso=64, split_size=[8,8], shift_size=[1,2], qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., rg_idx=0, b_idx=0):
        super().__init__()
        self.attn = Adaptive_Spatial_Attention(dim=channel, num_heads=num_heads, reso=reso, split_size=split_size, 
                                              shift_size=shift_size, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop, attn_drop=attn_drop, rg_idx=rg_idx, b_idx=b_idx)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_seq = rearrange(x, 'b c h w -> b (h w) c')
        out_seq = self.attn(x_seq, H, W)
        out = rearrange(out_seq, 'b (h w) c -> b c h w', h=H, w=W)
        return out


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试 SGFN
    print("=" * 60)
    print("测试 SGFN")
    print("=" * 60)
    input = torch.randn(1, 32, 256, 256)
    print(f"输入形状: {input.shape}")
    sgfn = SGFN_Wrapper(channel=32, hidden_features=128)
    sgfn.eval()  # 设置为评估模式，避免BatchNorm在batch_size=1时报错
    output = sgfn(input)
    print(f"输出形状: {output.shape}")
    assert output.shape == input.shape, f"形状不匹配: {output.shape} vs {input.shape}"
    print()
    
    # 测试 Adaptive_Channel_Attention
    print("=" * 60)
    print("测试 Adaptive_Channel_Attention")
    print("=" * 60)
    input = torch.randn(1, 32, 256, 256)
    print(f"输入形状: {input.shape}")
    channel_attn = Adaptive_Channel_Attention_Wrapper(channel=32, num_heads=8)
    channel_attn.eval()  # 设置为评估模式，避免BatchNorm在batch_size=1时报错
    output = channel_attn(input)
    print(f"输出形状: {output.shape}")
    assert output.shape == input.shape, f"形状不匹配: {output.shape} vs {input.shape}"
    print()
    
    # 测试 Adaptive_Spatial_Attention
    print("=" * 60)
    print("测试 Adaptive_Spatial_Attention")
    print("=" * 60)
    input = torch.randn(1, 32, 256, 256)
    print(f"输入形状: {input.shape}")
    spatial_attn = Adaptive_Spatial_Attention_Wrapper(channel=32, num_heads=8, reso=256, split_size=[8, 8], shift_size=[1, 2])
    spatial_attn.eval()  # 设置为评估模式，避免BatchNorm在batch_size=1时报错
    output = spatial_attn(input)
    print(f"输出形状: {output.shape}")
    assert output.shape == input.shape, f"形状不匹配: {output.shape} vs {input.shape}"
    print()
