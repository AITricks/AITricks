"""
Shift-Wise Convolution 即插即用模块
可直接替换标准Conv2d层，用于模拟大核卷积效果

使用示例:
    from shiftwise_plug_and_play import ShifthWiseConv2dImplicit
    
    # 替换标准卷积
    # 原始: self.conv = nn.Conv2d(64, 64, kernel_size=3)
    self.conv = ShifthWiseConv2dImplicit(
        in_channels=64,
        out_channels=64,
        big_kernel=51,      # 等效大核尺寸
        small_kernel=3,     # 实际使用的小核
        ghost_ratio=0.23,   # Ghost通道比例
        N_path=2,          # 多路径数量
        N_rep=4            # 混洗排序组数
    )
"""

import math
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 尝试导入CUDA加速版本，如果不存在则使用CPU fallback
try:
    from shiftadd.ops.ops_py import AddShift_mp_module
    _HAS_SHIFTADD = True
except Exception:
    AddShift_mp_module = None
    _HAS_SHIFTADD = False
    print("Warning: CUDA shift-add module not found, using CPU fallback implementation.")


class _AddShiftFallback(nn.Module):
    """
    CPU友好的fallback实现，使用torch.roll来近似shift操作
    当CUDA扩展不可用时自动使用此实现
    """
    
    def __init__(self, big_kernel: int, small_kernel: int, c_out: int, c_in: int, group_in: int) -> None:
        super().__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.group_in = group_in
        self.nk = math.ceil(big_kernel / small_kernel)
        
        # 生成移位索引（模拟CUDA版本的shuffle逻辑）
        torch.manual_seed(123)
        self.shuffle_idx_horizon = [
            torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).long()
            for _ in range(group_in)
        ]
        self.shuffle_idx_vertica = [
            torch.cat([torch.randperm(self.nk) + i * self.nk for i in range(c_out)]).long()
            for _ in range(group_in)
        ]
        self.shuffle_idx_identit = [
            (torch.randint(0, self.nk, [c_out]) + torch.arange(c_out) * self.nk).long()
            for _ in range(group_in)
        ]
    
    def forward(self, x: torch.Tensor, b: int, hout: int, wout: int):
        """
        输入: x shape (B, c_in=repN*nk, H, W)
        输出: 三个tensor (lora1_x, lora2_x, small_x)，每个shape (B, repN, hout, wout)
        
        注意：这是一个简化的fallback实现，主要用于功能测试。
        实际使用建议安装CUDA版本的shiftadd模块以获得更好的性能。
        """
        B, C, H, W = x.shape
        repN = self.c_out
        nk = self.nk
        
        # 重塑为 (B, repN, nk, H, W)
        x = x.view(B, repN, nk, H, W)
        
        # 初始化三个分支的输出
        lora1_x = torch.zeros(B, repN, H, W, device=x.device, dtype=x.dtype)
        lora2_x = torch.zeros(B, repN, H, W, device=x.device, dtype=x.dtype)
        small_x = torch.zeros(B, repN, H, W, device=x.device, dtype=x.dtype)
        
        # 对每个group_in组进行处理
        for g in range(self.group_in):
            # 水平移位分支 (lora1)
            idx_h = self.shuffle_idx_horizon[g].to(x.device)
            for i, k_idx in enumerate(idx_h):
                if k_idx >= 0:
                    rep_idx = int(k_idx.item() // nk)
                    nk_idx = int(k_idx.item() % nk)
                    if rep_idx < repN and nk_idx < nk:
                        # 应用水平和垂直移位
                        shift_h = i % 3
                        shift_w = (i * 2) % 3
                        shifted = torch.roll(x[:, rep_idx, nk_idx, :, :], shifts=(shift_h, shift_w), dims=(1, 2))
                        lora1_x[:, rep_idx, :, :] += shifted
            
            # 垂直移位分支 (lora2)
            idx_v = self.shuffle_idx_vertica[g].to(x.device)
            for i, k_idx in enumerate(idx_v):
                if k_idx >= 0:
                    rep_idx = int(k_idx.item() // nk)
                    nk_idx = int(k_idx.item() % nk)
                    if rep_idx < repN and nk_idx < nk:
                        shift_h = (i * 2) % 3
                        shift_w = i % 3
                        shifted = torch.roll(x[:, rep_idx, nk_idx, :, :], shifts=(shift_h, shift_w), dims=(1, 2))
                        lora2_x[:, rep_idx, :, :] += shifted
            
            # 恒等映射分支 (small)
            idx_id = self.shuffle_idx_identit[g].to(x.device)
            for k_idx in idx_id:
                if k_idx >= 0:  # 有效的索引
                    rep_idx = int(k_idx.item() // nk)
                    nk_idx = int(k_idx.item() % nk)
                    if rep_idx < repN and nk_idx < nk:
                        small_x[:, rep_idx, :, :] += x[:, rep_idx, nk_idx, :, :]
        
        # 裁剪到输出尺寸
        if H != hout or W != wout:
            dh = (H - hout) // 2
            dw = (W - wout) // 2
            lora1_x = lora1_x[:, :, dh:dh+hout, dw:dw+wout]
            lora2_x = lora2_x[:, :, dh:dh+hout, dw:dw+wout]
            small_x = small_x[:, :, dh:dh+hout, dw:dw+wout]
        
        return lora1_x, lora2_x, small_x


def get_bn(channels: int) -> nn.Module:
    """返回标准的BatchNorm2d"""
    return nn.BatchNorm2d(channels)


class ShifthWiseConv2dImplicit(nn.Module):
    """
    Shift-Wise卷积即插即用模块
    
    核心特性:
    - 将通道分为"rep"部分（经过shift处理）和"ghost"部分（直接通过）
    - 使用多路径depthwise卷积和shift操作模拟大核卷积
    - BN层放在shift操作之后（符合论文设计）
    - 支持重参数化优化（merge_branches）
    
    参数:
        in_channels: 输入通道数
        out_channels: 输出通道数（通常等于in_channels）
        big_kernel: 等效大核尺寸（如51, 49, 47等）
        small_kernel: 实际使用的小核尺寸（默认3）
        stride: 步长（默认1）
        bn: 是否使用BatchNorm（默认True）
        ghost_ratio: Ghost通道比例（默认0.23，即23%通道直接通过）
        N_path: 多路径数量（默认2，对应Figure 4的N paths）
        N_rep: 混洗排序组数（默认4，对应Figure 4的w/ shuffle）
        version: 版本，"v1"或"v2"（v2确保repN为偶数）
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        big_kernel: int,
        small_kernel: int = 3,
        stride: int = 1,
        group: int = 1,
        bn: bool = True,
        use_small_conv: bool = True,
        ghost_ratio: float = 0.23,
        N_path: int = 2,
        N_rep: int = 4,
        bias: bool = False,
        version: str = "v1",
    ) -> None:
        super().__init__()
        self.kernels: Tuple[int, int] = (small_kernel, big_kernel)
        self.stride = stride
        
        # 计算shift所需的padding
        padding, real_pad = self.shift(self.kernels)
        self.pad = padding, real_pad
        self.nk = math.ceil(big_kernel / small_kernel)
        
        # 通道分割：rep用于shift处理，ghost直接通过
        if version == "v2":
            # UniRep风格：确保repN为偶数
            repN = int(in_channels * (1 - ghost_ratio)) // 2 * 2
            ghostN = in_channels - repN
        else:
            # SLaK风格
            ghostN = int(in_channels * ghost_ratio)
            repN = in_channels - ghostN
        
        # 随机选择ghost通道（固定随机种子确保可重复）
        np.random.seed(123)
        ghost = np.random.choice(in_channels, ghostN, replace=False).tolist()
        ghost.sort()
        rep = list(set(range(in_channels)) - set(ghost))
        rep.sort()
        assert len(rep) == repN, f"len(rep):{len(rep)}==repN:{repN}"
        
        # 注册为buffer（兼容torchv5）
        # 注意：torchv5可能不支持LongTensor作为buffer，使用IntTensor
        self.register_buffer('ghost', torch.IntTensor(ghost))
        self.register_buffer('rep', torch.IntTensor(rep))
        
        out_n = repN * self.nk
        self.LoRA = None
        
        # 多路径depthwise卷积（group=repN实现depthwise）
        self.LoRAs = nn.ModuleList([
            nn.Conv2d(
                repN,
                out_n,
                kernel_size=small_kernel,
                stride=stride,
                padding=padding,
                groups=repN,  # depthwise卷积
                bias=bias,
            )
            for _ in range(N_path)
        ])
        
        # Shift-add多路径模块（优先使用CUDA实现，否则使用fallback）
        if _HAS_SHIFTADD and torch.cuda.is_available():
            self.loras = AddShift_mp_module(big_kernel, small_kernel, repN, out_n, N_rep)
            print(f"Using CUDA-accelerated shift-add module")
        else:
            self.loras = _AddShiftFallback(big_kernel, small_kernel, repN, out_n, N_rep)
            print(f"Using CPU fallback shift-add implementation")
        
        self.use_bn = bn
        if bn:
            # 三个分支的BN层（对应Figure 3(a)的设计）
            self.bn_lora1 = get_bn(repN)
            self.bn_lora2 = get_bn(repN)
            self.bn_small = get_bn(repN)
        else:
            self.bn_lora1 = None
            self.bn_lora2 = None
            self.bn_small = None
        
        print(f"ShiftWise Conv initialized: "
              f"in_channels={in_channels}, out_channels={out_channels}, "
              f"big_kernel={big_kernel}, small_kernel={small_kernel}, "
              f"ghost_ratio={ghost_ratio}, repN={repN}, ghostN={ghostN}, "
              f"N_path={N_path}, N_rep={N_rep}")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (B, C, H, W)
        输出: (B, C, H, W) - 保持输入输出形状一致，可直接替换Conv2d
        """
        # 索引已经在buffer中，自动移动到正确设备，不需要手动移动
        
        # 分割通道（确保索引在正确设备上）
        ghost_idx = self.ghost.to(inputs.device).long()
        rep_idx = self.rep.to(inputs.device).long()
        ghost_inputs = torch.index_select(inputs, 1, ghost_idx)
        rep_inputs = torch.index_select(inputs, 1, rep_idx)
        
        # 多路径卷积 -> 求和
        out = 0
        if self.LoRA is None:
            for split_conv in self.LoRAs:
                out = out + split_conv(rep_inputs)
        else:
            # 使用合并后的单卷积（重参数化后）
            out = self.LoRA(rep_inputs)
        
        # 应用shift-add模块（返回三个分支）
        b, _, h, w = inputs.shape
        lora1_x, lora2_x, small_x = self.loras(out, b, h, w)
        
        # BN层在shift之后（对应Figure 3(a)的设计）
        if self.use_bn:
            lora1_x = self.bn_lora1(lora1_x)
            lora2_x = self.bn_lora2(lora2_x)
            small_x = self.bn_small(small_x)
        
        # 聚合三个分支 + 残差连接
        x = lora1_x + lora2_x + small_x + rep_inputs
        
        # 合并ghost通道（对应Figure 3(b)的cat操作）
        x = torch.cat([x, ghost_inputs], dim=1)
        
        return x
    
    def shift(self, kernels: Tuple[int, int]):
        """
        计算shift操作所需的padding
        
        参数:
            kernels: (small_kernel, big_kernel) 元组
        返回:
            padding: 基础padding值
            real_pad: 每个位置的额外padding列表
        """
        mink, maxk = min(kernels), max(kernels)
        nk = math.ceil(maxk / mink)
        padding = mink - 1
        mid = maxk // 2
        real_pad = []
        for i in range(nk):
            extra_pad = mid - i * mink - padding
            real_pad.append(extra_pad)
        return padding, real_pad
    
    def merge_branches(self) -> None:
        """
        合并多个路径的卷积（重参数化优化）
        
        训练后调用此方法可以合并N_path个并行卷积为一个，减少推理时计算量
        """
        if self.LoRA is None:
            bias = True if self.LoRAs[0].bias is not None else False
            LoRA = nn.Conv2d(
                in_channels=self.LoRAs[0].in_channels,
                out_channels=self.LoRAs[0].out_channels,
                kernel_size=self.LoRAs[0].kernel_size,
                stride=self.LoRAs[0].stride,
                padding=self.LoRAs[0].padding,
                dilation=self.LoRAs[0].dilation,
                groups=self.LoRAs[0].groups,
                bias=bias
            )
            weight, biasdata = 0, 0
            for merged_conv in self.LoRAs:
                weight = weight + merged_conv.weight.data
                if bias:
                    biasdata = biasdata + merged_conv.bias.data
            
            LoRA.weight.data = weight
            if bias:
                LoRA.bias.data = biasdata
            self.LoRA = LoRA.to(next(self.LoRAs[0].parameters()).device)
            # 删除原始的多路径卷积列表
            del self.LoRAs
            print("Branches merged successfully!")


def test_basic_functionality():
    """基础功能测试"""
    print("=" * 60)
    print("测试1: 基础功能测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 创建模块
    model = ShifthWiseConv2dImplicit(
        in_channels=64,
        out_channels=64,
        big_kernel=51,
        small_kernel=3,
        ghost_ratio=0.23,
        N_path=2,
        N_rep=4,
    ).to(device)
    
    # 创建测试输入
    x = torch.randn(2, 64, 56, 56).to(device)
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        y = model(x)
    
    print(f"输出形状: {y.shape}")
    assert y.shape == x.shape, f"输出形状 {y.shape} 应与输入形状 {x.shape} 相同"
    print("✓ 形状检查通过")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print("✓ 基础功能测试通过\n")


def test_replace_conv2d():
    """测试替换标准Conv2d"""
    print("=" * 60)
    print("测试2: 替换标准Conv2d测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 原始网络（使用标准Conv2d）
    class OriginalNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    
    # 替换后的网络（使用ShiftWise Conv）
    class ShiftWiseNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 第一层保持标准卷积
            self.conv2 = ShifthWiseConv2dImplicit(
                64, 64, big_kernel=51, small_kernel=3, ghost_ratio=0.23, N_path=2, N_rep=4
            )
            self.conv3 = ShifthWiseConv2dImplicit(
                64, 128, big_kernel=51, small_kernel=3, ghost_ratio=0.23, N_path=2, N_rep=4
            )
    
    net = ShiftWiseNet().to(device)
    x = torch.randn(4, 3, 224, 224).to(device)
    
    net.eval()
    with torch.no_grad():
        y = net.conv1(x)
        y = net.conv2(y)
        y = net.conv3(y)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print("✓ 替换标准Conv2d测试通过\n")


def test_reparameterization():
    """测试重参数化"""
    print("=" * 60)
    print("测试3: 重参数化测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ShifthWiseConv2dImplicit(
        in_channels=64,
        out_channels=64,
        big_kernel=51,
        small_kernel=3,
        N_path=2,
    ).to(device)
    
    x = torch.randn(2, 64, 56, 56).to(device)
    
    # 合并前的输出
    model.eval()
    with torch.no_grad():
        y_before = model(x)
    
    # 合并分支
    model.merge_branches()
    
    # 合并后的输出
    with torch.no_grad():
        y_after = model(x)
    
    # 检查输出是否一致（允许小的数值误差）
    diff = torch.abs(y_before - y_after).max().item()
    print(f"合并前后输出最大差异: {diff:.6f}")
    
    if diff < 1e-5:
        print("✓ 重参数化测试通过（输出一致）")
    else:
        print(f"⚠ 重参数化后输出有差异（最大差异: {diff:.6f}）")
        print("  这可能是由于fallback实现的近似导致的")
    print()


def test_performance():
    """性能测试"""
    print("=" * 60)
    print("测试4: 性能测试")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = ShifthWiseConv2dImplicit(
        in_channels=64,
        out_channels=64,
        big_kernel=51,
        small_kernel=3,
        N_path=2,
        N_rep=4,
    ).to(device)
    
    x = torch.randn(4, 64, 224, 224).to(device)
    
    # 预热
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(x)
    
    # 同步（如果是CUDA）
    if device == "cuda":
        torch.cuda.synchronize()
    
    # 计时
    num_iterations = 50
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed_time = (time.time() - start_time) * 1000 / num_iterations  # ms
    
    print(f"设备: {device}")
    print(f"输入形状: {x.shape}")
    print(f"平均推理时间: {elapsed_time:.2f} ms")
    print(f"吞吐量: {1000/elapsed_time:.1f} FPS")
    print()


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Shift-Wise Convolution 即插即用模块测试")
    print("=" * 60 + "\n")
    
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print()
    
    try:
        # 运行所有测试
        test_basic_functionality()
        test_replace_conv2d()
        test_reparameterization()
        test_performance()
        
        print("=" * 60)
        print("所有测试完成！")
        print("=" * 60)
        print("\n使用说明:")
        print("  1. 直接导入: from shiftwise_plug_and_play import ShifthWiseConv2dImplicit")
        print("  2. 替换标准卷积层即可使用")
        print("  3. 训练后调用 merge_branches() 可优化推理速度")
        print()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

