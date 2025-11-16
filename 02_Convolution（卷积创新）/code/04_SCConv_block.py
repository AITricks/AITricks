import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupBatchnorm2d(nn.Module):
    """
    二维组归一化层 (Group Normalization)
    根据论文，使用组归一化来处理特征图
    """
    def __init__(self, c_num: int, group_num: int = 16, eps: float = 1e-10):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num, f"通道数 {c_num} 必须大于等于组数 {group_num}"
        self.group_num = group_num
        # gamma参数衡量特征图中空间信息的不同，空间信息越丰富，gamma越大
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        # 重新塑形以便按组计算均值和标准差
        x = x.view(N, self.group_num, -1)
        
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        
        # 恢复原始形状并应用缩放和平移
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    """
    Spatial Reconstruction Unit (空间重构单元)
    
    根据图2的架构：
    1. 输入X经过Group Normalization得到gn_x
    2. 从gamma计算权重 w_i = γ_i / Σ_j γ_j
    3. 计算重加权值 reweights = Sigmoid(gn_x * w_gamma)
    4. 通过阈值分离出信息量大和信息量少的特征图
    5. 交叉重构得到最终输出
    """
    def __init__(self, oup_channels: int, group_num: int = 16, gate_threshold: float = 0.5):
        super().__init__()
        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Step 1: Group Normalization
        # GN层的可训练参数γ衡量特征图中空间信息的不同，空间信息越是丰富，γ越大
        gn_x = self.gn(x)
        
        # Step 2: 计算权重化的gamma
        # w_i = γ_i / Σ_j γ_j，归一化每个通道的gamma值
        w_gamma = self.gn.gamma.view(1, -1, 1, 1) / torch.sum(self.gn.gamma)
        
        # Step 3: 计算重加权值并应用Sigmoid
        # 根据图2，应该是先相乘w_i，然后Sigmoid，然后Threshold
        reweights = self.sigmoid(gn_x * w_gamma)
        
        # Step 4: 门控机制，通过阈值分离信息
        # 获得信息量大和信息量较少的两个特征图
        info_mask = reweights >= self.gate_threshold
        noninfo_mask = reweights < self.gate_threshold
        
        x_1 = info_mask.float() * x  # 信息丰富的部分
        x_2 = noninfo_mask.float() * x  # 信息较少的部分
        
        # Step 5: 重构输出
        # 根据图2，交叉相乘与cat，获得最终的输出特征
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        """
        重构方法：交叉相加
        根据图2：X11^W + X22^W 和 X12^W + X21^W，然后concatenate
        """
        # 将x_1和x_2各分成两半
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        
        # 交叉相加并拼接：能够更加有效地联合两个特征并且加强特征之间的交互
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    """
    Channel Reconstruction Unit (通道重构单元)
    
    根据图3的架构，分为三个阶段：
    1. Split: 使用1x1 Conv将输入分成Xup和Xlow
    2. Transform: 分别对Xup和Xlow进行变换
    3. Fuse: 使用attention机制融合Y1和Y2
    """
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,  # 分割比例，0 < alpha < 1
                 squeeze_radio: int = 2,  # 压缩率
                 group_size: int = 2,  # 组卷积的组大小
                 group_kernel_size: int = 3  # 组卷积核大小
                 ):
        super().__init__()
        
        self.up_channel = int(alpha * op_channel)  # 上半部分通道数
        self.low_channel = op_channel - self.up_channel  # 下半部分通道数
        
        # Split阶段：使用1x1 Conv来分割通道（根据图3）
        self.split_conv_up = nn.Conv2d(op_channel, self.up_channel, kernel_size=1, bias=False)
        self.split_conv_low = nn.Conv2d(op_channel, self.low_channel, kernel_size=1, bias=False)
        
        # Transform阶段的压缩卷积
        self.squeeze1 = nn.Conv2d(self.up_channel, self.up_channel // squeeze_radio, 
                                  kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(self.low_channel, self.low_channel // squeeze_radio, 
                                  kernel_size=1, bias=False)
        
        # Transform阶段：上半部分（Upper branch）
        # GWC: Group-wise Convolution
        self.GWC = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, 
                            kernel_size=group_kernel_size, stride=1,
                            padding=group_kernel_size // 2, groups=group_size, bias=False)
        # PWC: Point-wise Convolution
        self.PWC1 = nn.Conv2d(self.up_channel // squeeze_radio, op_channel, 
                             kernel_size=1, bias=False)
        
        # Transform阶段：下半部分（Lower branch）
        # PWC处理low_channel的一部分，然后与原始low拼接
        self.PWC2 = nn.Conv2d(self.low_channel // squeeze_radio, 
                             op_channel - self.low_channel // squeeze_radio, 
                             kernel_size=1, bias=False)
        
        # Fuse阶段：自适应平均池化（用于生成attention权重）
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        """
        根据图3的流程：
        1. Split: 将输入分成Xup和Xlow
        2. Transform: 分别变换得到Y1和Y2
        3. Fuse: 使用attention融合Y1和Y2
        """
        # ========== Split阶段 ==========
        # 根据图3，使用1x1 Conv来分割
        xup = self.split_conv_up(x)  # αC channels
        xlow = self.split_conv_low(x)  # (1-α)C channels
        
        # ========== Transform阶段 ==========
        # 压缩
        up = self.squeeze1(xup)
        low = self.squeeze2(xlow)
        
        # Upper branch: GWC + PWC，然后element-wise summation
        Y1 = self.GWC(up) + self.PWC1(up)
        
        # Lower branch: PWC + 直接通道，然后concatenation
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        
        # ========== Fuse阶段 ==========
        # 根据图3，应该分别对Y1和Y2做Pooling得到S1和S2，然后concat，再softmax得到β1和β2
        # 然后 Y1*β1 + Y2*β2
        
        # 对Y1和Y2分别做全局平均池化，得到S1和S2
        S1 = self.adaptive_avg_pool(Y1)  # [B, C, 1, 1] where C = op_channel
        S2 = self.adaptive_avg_pool(Y2)  # [B, C, 1, 1] where C = op_channel
        
        # 根据图3：S1和S2应该都是op_channel通道，然后concat得到[B, 2C, 1, 1]
        # 然后softmax得到两个分支的attention权重
        S_concat = torch.cat([S1, S2], dim=1)  # [B, 2C, 1, 1]
        
        # 将S_concat reshape为 [B, 2, C, 1, 1]，这样可以在dim=1上应用softmax
        # 得到每个分支的通道级attention权重
        B, C = S1.size(0), S1.size(1)
        S_reshaped = S_concat.view(B, 2, C, 1, 1)  # [B, 2, C, 1, 1]
        beta = F.softmax(S_reshaped, dim=1)  # [B, 2, C, 1, 1]，在分支维度上softmax
        
        # 提取β1和β2
        beta1 = beta[:, 0, :, :, :]  # [B, C, 1, 1]
        beta2 = beta[:, 1, :, :, :]  # [B, C, 1, 1]
        
        # Element-wise multiplication and summation
        # Y1 * β1 + Y2 * β2
        out = Y1 * beta1 + Y2 * beta2
        
        return out


class SCConv(nn.Module):
    """
    SCConv: Spatial and Channel reconstruction Convolution
    即插即用的卷积模块，集成了SRU和CRU
    
    根据图1的架构：
    输入 -> SRU -> CRU -> 输出
    """
    def __init__(self,
                 op_channel: int,  # 操作通道数量
                 group_num: int = 16,  # Group Normalization的组数
                 gate_threshold: float = 0.5,  # SRU的阈值
                 alpha: float = 1 / 2,  # CRU的分割比例
                 squeeze_radio: int = 2,  # CRU的压缩率
                 group_size: int = 2,  # CRU的组卷积组大小
                 group_kernel_size: int = 3  # CRU的组卷积核大小
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                      group_num=group_num,
                      gate_threshold=gate_threshold)
        self.CRU = CRU(op_channel,
                      alpha=alpha,
                      squeeze_radio=squeeze_radio,
                      group_size=group_size,
                      group_kernel_size=group_kernel_size)

    def forward(self, x):
        """
        前向传播：先经过SRU进行空间重构，再经过CRU进行通道重构
        """
        x = self.SRU(x)  # 空间重构
        x = self.CRU(x)  # 通道重构
        return x


if __name__ == '__main__':
    # 测试代码
    print("=" * 50)
    print("测试 SCConv 模块")
    print("=" * 50)
    
    # 创建测试输入
    batch_size = 2
    channels = 64
    height, width = 128, 128
    
    input_tensor = torch.randn(batch_size, channels, height, width)
    print(f"输入形状: {input_tensor.shape}")
    
    # 创建SCConv模块
    model = SCConv(op_channel=channels,
                   group_num=16,
                   gate_threshold=0.5,
                   alpha=1/2,
                   squeeze_radio=2,
                   group_size=2,
                   group_kernel_size=3)
    
    # 前向传播
    output = model(input_tensor)
    print(f"输出形状: {output.shape}")
    
    # 验证输入输出通道数是否一致
    assert input_tensor.shape[1] == output.shape[1], \
        f"通道数不匹配！输入: {input_tensor.shape[1]}, 输出: {output.shape[1]}"
    assert input_tensor.shape[2] == output.shape[2], \
        f"高度不匹配！输入: {input_tensor.shape[2]}, 输出: {output.shape[2]}"
    assert input_tensor.shape[3] == output.shape[3], \
        f"宽度不匹配！输入: {input_tensor.shape[3]}, 输出: {output.shape[3]}"
    
    print("\n✓ 所有测试通过！模块工作正常。")
    print("=" * 50)
