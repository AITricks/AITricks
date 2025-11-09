"""
LDConv (Learnable Deformable Convolution) - 可学习变形卷积模块
基于AKConv原理实现，支持任意采样形状卷积核与任意参数数量卷积核

核心思想：
1. 通过Conv2d生成可学习的offsets（偏移量）
2. 将offsets应用到初始采样坐标上，得到修改后的采样坐标
3. 基于修改后的坐标进行双线性插值重采样
4. 将重采样后的特征进行reshape和卷积操作

对应结构图流程：
Input → Conv2d(生成Offset) → 初始采样坐标(p_n) → 坐标修改(p = p_0 + p_n + offset) 
→ Resample(基于修改坐标重采样) → Bilinear Interpolation → Reshape → Conv + BN + SiLU → Output
"""

import torch
import torch.nn as nn
import math
from einops import rearrange


class LDConv(nn.Module):
    """
    LDConv: 可学习变形卷积模块，即插即用替换普通卷积
    
    Args:
        inc (int): 输入通道数
        outc (int): 输出通道数
        num_param (int): 采样点数量（卷积核参数数量），例如9表示3x3网格
        stride (int): 步长，默认为1
        bias (bool): 是否使用偏置，默认为None
    """
    def __init__(self, inc, outc, num_param, stride=1, bias=None):
        super(LDConv, self).__init__()
        self.num_param = num_param  # 采样点数量（对应结构图中的N）
        self.stride = stride
        
        # 最终卷积层：对应结构图中的 Conv + Norm + SiLU
        # 使用列方向卷积：kernel_size=(num_param, 1), stride=(num_param, 1)
        # 这样可以有效地处理重采样后的特征
        self.conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=(num_param, 1), stride=(num_param, 1), bias=bias),
            nn.BatchNorm2d(outc),
            nn.SiLU()
        )
        
        # 偏移量生成卷积：对应结构图中的 Conv2d → Offset
        # 输出通道数为 2 * num_param（每个采样点需要x和y两个方向的偏移）
        self.p_conv = nn.Conv2d(inc, 2 * num_param, kernel_size=3, padding=1, stride=stride)
        # 初始化偏移量卷积权重为0，确保初始时采样点位置接近规则网格
        nn.init.constant_(self.p_conv.weight, 0)
        # 注册反向传播钩子，降低偏移量学习的学习率，使其更稳定
        self.p_conv.register_full_backward_hook(self._set_lr)
        
        # 注册初始采样坐标：对应结构图中的 Initial Sampling Points (p_n)
        # 这些是固定的初始采样位置，会根据num_param自动生成
        self.register_buffer("p_n", self._get_p_n(N=self.num_param))

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        """
        反向传播钩子函数：降低偏移量学习的梯度
        使偏移量的学习更加稳定，避免训练初期采样点位置变化过大
        """
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x (Tensor): 输入特征图，形状为 (B, C, H, W)
            
        Returns:
            out (Tensor): 输出特征图，形状为 (B, outc, H', W')
            
        流程说明（对应结构图）：
        1. 生成Offset：通过p_conv生成可学习的偏移量
        2. 坐标修改：p = p_0 + p_n + offset（基础坐标 + 初始采样点 + 偏移量）
        3. 双线性插值：计算四个最近邻点的插值权重
        4. 重采样：基于修改后的坐标从输入特征图中采样
        5. Reshape：将重采样后的特征重塑为卷积层可处理的形状
        6. Conv：最终卷积、归一化和激活
        """
        # 步骤1：生成Offset（对应结构图中的 Conv2d → Offset）
        # offset形状: (b, 2N, h, w)，其中N=num_param，2N表示每个采样点的x和y偏移
        offset = self.p_conv(x)
        dtype = offset.data.type()
        N = offset.size(1) // 2  # N是采样点数量
        
        # 步骤2：计算修改后的采样坐标（对应结构图中的坐标修改）
        # p形状: (b, 2N, h, w)，包含所有采样点的修改后坐标
        # p = p_0（基础坐标网格） + p_n（初始采样点） + offset（可学习偏移）
        p = self._get_p(offset, dtype)

        # 步骤3：准备双线性插值（对应结构图中的 Bilinear Interpolation）
        # 将坐标从 (b, 2N, h, w) 转换为 (b, h, w, 2N) 以便后续处理
        p = p.contiguous().permute(0, 2, 3, 1)
        
        # 计算四个最近邻整数坐标点（左上、右下、左下、右上）
        q_lt = p.detach().floor()  # 左上角（left-top）
        q_rb = q_lt + 1            # 右下角（right-bottom）
        
        # 将坐标限制在特征图范围内，防止越界
        q_lt = torch.cat([
            torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),  # x坐标限制
            torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)   # y坐标限制
        ], dim=-1).long()
        q_rb = torch.cat([
            torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
            torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)
        ], dim=-1).long()
        
        # 计算左下和右上角坐标
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)  # 左下（left-bottom）
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)  # 右上（right-top）

        # 同样将实际采样坐标p限制在特征图范围内
        p = torch.cat([
            torch.clamp(p[..., :N], 0, x.size(2) - 1),
            torch.clamp(p[..., N:], 0, x.size(3) - 1)
        ], dim=-1)

        # 步骤4：计算双线性插值权重（对应结构图中的 Bilinear Kernel）
        # g_lt, g_rb, g_lb, g_rt 形状: (b, h, w, N)
        # 权重计算公式基于双线性插值的标准公式
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 步骤5：基于修改后的坐标重采样（对应结构图中的 Resample）
        # 从输入特征图中提取四个最近邻点的特征值
        x_q_lt = self._get_x_q(x, q_lt, N)  # 左上角特征
        x_q_rb = self._get_x_q(x, q_rb, N)  # 右下角特征
        x_q_lb = self._get_x_q(x, q_lb, N)  # 左下角特征
        x_q_rt = self._get_x_q(x, q_rt, N)  # 右上角特征

        # 步骤6：执行双线性插值（对应结构图中的 Bilinear Interpolation）
        # 使用权重对四个最近邻点进行加权求和
        # x_offset形状: (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # 步骤7：Reshape重采样后的特征（对应结构图中的 Reshape）
        # 将形状从 (b, c, h, w, N) 重塑为 (b, c, h*N, w)
        x_offset = self._reshape_x_offset(x_offset, self.num_param)
        
        # 步骤8：最终卷积、归一化和激活（对应结构图中的 Conv + Norm + SiLU）
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N):
        """
        生成初始采样坐标（对应结构图中的 Initial Sampling Points）
        
        根据num_param生成规则的初始采样网格。例如：
        - num_param=9  → 3x3网格
        - num_param=16 → 4x4网格
        - num_param=25 → 5x5网格
        
        Args:
            N (int): 采样点数量（等于num_param）
            
        Returns:
            p_n (Tensor): 初始采样坐标，形状为 (1, 2N, 1, 1)
                         前N个通道是x坐标，后N个通道是y坐标
        """
        # 计算基础网格大小（接近sqrt(num_param)的整数）
        base_int = round(math.sqrt(self.num_param))
        row_number = self.num_param // base_int  # 行数
        mod_number = self.num_param % base_int    # 余数（处理非完全平方数的情况）
        
        # 生成主要的网格坐标
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(0, row_number),      # x坐标：0到row_number-1
            torch.arange(0, base_int),        # y坐标：0到base_int-1
            indexing='ij'  # 使用矩阵索引模式，确保与原始代码行为一致
        )
        p_n_x = torch.flatten(p_n_x)  # 展平为1D
        p_n_y = torch.flatten(p_n_y)  # 展平为1D
        
        # 如果num_param不是完全平方数，添加额外的采样点
        if mod_number > 0:
            mod_p_n_x, mod_p_n_y = torch.meshgrid(
                torch.arange(row_number, row_number + 1),  # 额外的一行
                torch.arange(0, mod_number),               # 额外的列
                indexing='ij'  # 使用矩阵索引模式
            )
            mod_p_n_x = torch.flatten(mod_p_n_x)
            mod_p_n_y = torch.flatten(mod_p_n_y)
            # 拼接主要网格和额外点
            p_n_x = torch.cat((p_n_x, mod_p_n_x))
            p_n_y = torch.cat((p_n_y, mod_p_n_y))
        
        # 将x和y坐标拼接，形状: (2N,)
        p_n = torch.cat([p_n_x, p_n_y], 0)
        # 重塑为 (1, 2N, 1, 1)，便于后续广播
        p_n = p_n.view(1, 2 * N, 1, 1)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        """
        生成基础坐标网格（无零填充）
        
        为输出特征图的每个位置生成基础坐标网格。
        这些坐标表示输出特征图中每个像素对应的输入特征图中的位置。
        
        Args:
            h (int): 输出特征图的高度
            w (int): 输出特征图的宽度
            N (int): 采样点数量
            dtype: 数据类型
            
        Returns:
            p_0 (Tensor): 基础坐标网格，形状为 (1, 2N, h, w)
                         前N个通道是x坐标，后N个通道是y坐标
        """
        # 生成输出特征图的坐标网格
        # 考虑stride，坐标间隔为stride
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(0, h * self.stride, self.stride),  # x坐标
            torch.arange(0, w * self.stride, self.stride),  # y坐标
            indexing='ij'  # 使用矩阵索引模式，确保与原始代码行为一致
        )
        
        # 展平并重塑为 (1, 1, h, w)，然后复制N次
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        
        # 拼接x和y坐标，形状: (1, 2N, h, w)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        """
        计算修改后的采样坐标（对应结构图中的坐标修改机制）
        
        最终采样坐标 = 基础坐标网格 + 初始采样点 + 可学习偏移量
        p = p_0 + p_n + offset
        
        Args:
            offset (Tensor): 可学习的偏移量，形状为 (b, 2N, h, w)
            dtype: 数据类型
            
        Returns:
            p (Tensor): 修改后的采样坐标，形状为 (b, 2N, h, w)
        """
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)

        # p_n: 初始采样点坐标，形状 (1, 2N, 1, 1)
        # p_0: 基础坐标网格，形状 (1, 2N, h, w)
        # offset: 可学习偏移量，形状 (b, 2N, h, w)
        # 通过广播机制相加，得到最终采样坐标
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + self.p_n + offset  # 坐标修改：基础坐标 + 初始采样点 + 偏移量
        return p

    def _get_x_q(self, x, q, N):
        """
        基于坐标q从输入特征图x中采样特征（对应结构图中的 Resample）
        
        使用gather操作根据坐标索引从输入特征图中提取特征值。
        这是双线性插值的第一步，获取四个最近邻点的特征值。
        
        Args:
            x (Tensor): 输入特征图，形状为 (b, c, H, W)
            q (Tensor): 采样坐标（整数坐标），形状为 (b, h, w, 2N)
                       前N个通道是x坐标，后N个通道是y坐标
            N (int): 采样点数量
            
        Returns:
            x_offset (Tensor): 采样得到的特征，形状为 (b, c, h, w, N)
        """
        b, h, w, _ = q.size()
        padded_w = x.size(3)  # 输入特征图的宽度
        c = x.size(1)         # 输入特征图的通道数
        
        # 将输入特征图重塑为 (b, c, h*w)，便于索引
        x = x.contiguous().view(b, c, -1)

        # 计算线性索引：将2D坐标 (x, y) 转换为1D索引
        # index形状: (b, h, w, N)
        index = q[..., :N] * padded_w + q[..., N:]  # index = x * W + y
        
        # 扩展索引维度以匹配特征图的通道数
        # index形状: (b, c, h, w, N) → (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        # 使用gather从输入特征图中提取对应位置的特征值
        # x_offset形状: (b, c, h, w, N)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, num_param):
        """
        重塑重采样后的特征（对应结构图中的 Reshape）
        
        将重采样后的特征从 (b, c, h, w, n) 重塑为 (b, c, h*n, w)，
        以便后续使用列方向卷积处理。这种重塑方式将N个采样点的特征
        按行方向堆叠，然后使用 kernel_size=(num_param, 1) 的卷积
        在列方向上进行聚合。
        
        Args:
            x_offset (Tensor): 重采样后的特征，形状为 (b, c, h, w, n)
            num_param (int): 采样点数量
            
        Returns:
            x_offset (Tensor): 重塑后的特征，形状为 (b, c, h*n, w)
            
        注意：这里使用了三种可选的实现方式：
        1. Conv3d方式：使用3D卷积处理
        2. 1×1 Conv方式：展平通道维度后使用1×1卷积
        3. 列方向Conv方式（当前实现）：按行堆叠后使用列方向卷积
        """
        b, c, h, w, n = x_offset.size()
        
        # 使用einops的rearrange将形状从 (b, c, h, w, n) 重塑为 (b, c, h*n, w)
        # 这相当于将N个采样点的特征按行方向堆叠
        # 例如：h=32, n=9 → 重塑后高度变为 32*9=288
        x_offset = rearrange(x_offset, 'b c h w n -> b c (h n) w')
        return x_offset


def main():
    """
    LDConv即插即用模块演示
    """
    print("=== LDConv即插即用模块演示 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    
    # 模拟输入特征图
    x = torch.randn(batch_size, channels, height, width).to(device)
    print(f"输入张量形状: {x.shape}")
    
    # 创建LDConv模块 - 即插即用替换普通卷积
    ldconv = LDConv(
        inc=channels,      # 输入通道数
        outc=128,          # 输出通道数  
        num_param=9,       # 采样点数量 (3x3网格)
        stride=1,          # 步长
        bias=False         # 是否使用偏置
    ).to(device)
    
    print(f"LDConv模块参数:")
    print(f"  输入通道: {channels}")
    print(f"  输出通道: 128")
    print(f"  采样点数量: 9")
    print(f"  步长: 1")
    
    # 前向传播
    print("\n开始前向传播...")
    with torch.no_grad():
        output = ldconv(x)
    
    print(f"输出张量形状: {output.shape}")
    print(f"输入输出尺寸变化: {x.shape} -> {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in ldconv.parameters())
    print(f"模块总参数量: {total_params:,}")
    
    # 与普通卷积对比
    print("\n=== 与普通卷积对比 ===")
    normal_conv = nn.Conv2d(channels, 128, kernel_size=3, padding=1, stride=1, bias=False).to(device)
    normal_params = sum(p.numel() for p in normal_conv.parameters())
    
    print(f"普通Conv2d参数量: {normal_params:,}")
    print(f"LDConv参数量: {total_params:,}")
    print(f"参数量增加: {((total_params - normal_params) / normal_params * 100):.1f}%")
    
    # 测试不同采样点数量
    print("\n=== 测试不同采样点数量 ===")
    for num_points in [4, 9, 16, 25]:
        test_conv = LDConv(channels, 128, num_points, stride=1, bias=False).to(device)
        test_params = sum(p.numel() for p in test_conv.parameters())
        print(f"采样点数量 {num_points}: 参数量 {test_params:,}")
    
    # 即插即用替换演示
    print("\n=== 即插即用替换演示 ===")
    
    # 创建一个简单的CNN网络
    class SimpleCNN(nn.Module):
        def __init__(self, use_ldconv=True):
            super().__init__()
            if use_ldconv:
                # 使用LDConv替换普通卷积
                self.conv1 = LDConv(3, 64, 9, stride=2)
                self.conv2 = LDConv(64, 128, 9, stride=2)
                self.conv3 = LDConv(128, 256, 9, stride=2)
                print("使用LDConv模块")
            else:
                # 普通卷积
                self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
                self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1) 
                self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
                print("使用普通Conv2d模块")
            
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    # 测试两种网络
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    # 普通CNN
    normal_cnn = SimpleCNN(use_ldconv=False).to(device)
    with torch.no_grad():
        normal_output = normal_cnn(test_input)
    
    # LDConv CNN  
    ldconv_cnn = SimpleCNN(use_ldconv=True).to(device)
    with torch.no_grad():
        ldconv_output = ldconv_cnn(test_input)
    
    print(f"普通CNN输出形状: {normal_output.shape}")
    print(f"LDConv CNN输出形状: {ldconv_output.shape}")
    
    normal_params = sum(p.numel() for p in normal_cnn.parameters())
    ldconv_params = sum(p.numel() for p in ldconv_cnn.parameters())
    
    print(f"普通CNN参数量: {normal_params:,}")
    print(f"LDConv CNN参数量: {ldconv_params:,}")
    
    print("\n=== 演示完成 ===")
    print("LDConv模块可以无缝替换普通卷积，实现即插即用！")


if __name__ == "__main__":
    main()