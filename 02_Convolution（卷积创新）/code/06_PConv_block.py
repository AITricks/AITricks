import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """
    自动计算padding以保持输出shape与输入相同
    Args:
        k: 卷积核大小
        p: padding大小（如果为None则自动计算）
        d: 空洞卷积的dilation
    """
    if d > 1:
        # 计算实际的卷积核大小（考虑dilation）
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # 自动计算padding（保持same shape）
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """
    标准卷积模块：Conv2d + BatchNorm + Activation
    包含卷积、批归一化和激活函数的标准组合
    """

    default_act = nn.SiLU()  # 默认激活函数为SiLU

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        初始化卷积层
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小
            s: 步长
            p: padding
            g: 分组卷积的组数
            d: 空洞卷积的dilation
            act: 激活函数（True使用默认SiLU，False不使用，或传入自定义激活函数）
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播：卷积 -> BN -> 激活"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """融合模式的前向传播（跳过BN）"""
        return self.act(self.conv(x))


class PConv(nn.Module):  
    """
    风车形卷积（Pinwheel-shaped Convolution）
    使用非对称填充方法实现风车形的感受野
    
    核心思想：
    1. 对输入特征图进行4个方向的非对称padding
    2. 使用水平(1,k)和垂直(k,1)卷积分别提取特征
    3. 将4个方向的特征concat后融合
    4. 形成类似风车的感受野，提升特征提取能力
    """
    
    def __init__(self, c1, c2, k, s):
        """
        初始化风车形卷积
        Args:
            c1: 输入通道数
            c2: 输出通道数
            k: 卷积核大小（用于非对称padding）
            s: 步长
        """
        super().__init__()

        # 定义4个方向的非对称padding: (left, right, top, bottom)
        # 分别对应：右、左、下、上四个方向的扩展
        p = [(k, 0, 1, 0), (0, k, 0, 1), (0, 1, k, 0), (1, 0, 0, k)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        
        # 水平方向卷积 (1, k)：捕获水平方向特征
        self.cw = Conv(c1, c2 // 4, (1, k), s=s, p=0)
        # 垂直方向卷积 (k, 1)：捕获垂直方向特征
        self.ch = Conv(c1, c2 // 4, (k, 1), s=s, p=0)
        # 融合卷积：将4个方向的特征融合
        self.cat = Conv(c2, c2, 2, s=1, p=0)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            融合后的特征图 [B, C2, H', W']
        """
        # 水平方向：左右两个方向的特征
        yw0 = self.cw(self.pad[0](x))  # 右侧padding + 水平卷积
        yw1 = self.cw(self.pad[1](x))  # 左侧padding + 水平卷积
        
        # 垂直方向：上下两个方向的特征
        yh0 = self.ch(self.pad[2](x))  # 下侧padding + 垂直卷积
        yh1 = self.ch(self.pad[3](x))  # 上侧padding + 垂直卷积
        
        # 将4个方向的特征concat后融合
        return self.cat(torch.cat([yw0, yw1, yh0, yh1], dim=1))


class APC2f(nn.Module):
    """
    APC2f模块：基于非对称填充的CSP瓶颈结构
    APCSP (Asymmetric Padding CSP) Bottleneck的快速实现
    """

    def __init__(self, c1, c2, n=1, shortcut=False, P=True, g=1, e=0.5):
        """
        初始化APC2f模块
        Args:
            c1: 输入通道数
            c2: 输出通道数
            n: bottleneck重复次数
            shortcut: 是否使用残差连接
            P: True使用APBottleneck，False使用标准Bottleneck
            g: 分组卷积的组数
            e: 通道扩展系数
        """
        super().__init__()
        self.c = int(c2 * e)  # 隐藏层通道数
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 初始卷积，将通道扩展为2倍
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # 最终融合卷积
        
        # 根据P参数选择使用APBottleneck或标准Bottleneck
        if P:
            # 使用非对称填充的Bottleneck
            self.m = nn.ModuleList(APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        else:
            # 注意：标准Bottleneck需要从其他模块导入（如YOLO的common模块）
            # 这里暂时使用APBottleneck替代，实际使用时请根据需求导入对应的Bottleneck
            self.m = nn.ModuleList(APBottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """
        前向传播（CSP结构）
        Args:
            x: 输入特征图
        Returns:
            输出特征图
        """
        # 将输入分成两部分
        y = list(self.cv1(x).chunk(2, 1))
        # 对第二部分进行多次bottleneck处理，并将每次的输出添加到列表中
        y.extend(m(y[-1]) for m in self.m)
        # 将所有特征concat后融合
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """使用split()代替chunk()的前向传播"""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class APBottleneck(nn.Module):
    """
    非对称填充瓶颈模块（Asymmetric Padding Bottleneck）
    通过4个方向的非对称padding提取多方向特征
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """
        初始化APBottleneck
        Args:
            c1: 输入通道数
            c2: 输出通道数
            shortcut: 是否使用残差连接
            g: 分组卷积的组数
            k: 卷积核大小元组
            e: 通道扩展系数
        """
        super().__init__()
        c_ = int(c2 * e)  # 隐藏层通道数
        
        # 定义4个方向的非对称padding: (left, right, top, bottom)
        p = [(2,0,2,0),(0,2,0,2),(0,2,2,0),(2,0,0,2)]
        self.pad = [nn.ZeroPad2d(padding=(p[g])) for g in range(4)]
        
        # 第一层卷积：每个方向输出c_//4通道
        self.cv1 = Conv(c1, c_ // 4, k[0], 1, p=0)
        # 第二层卷积：融合4个方向的特征
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # 判断是否可以使用残差连接（输入输出通道数相同）
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图
        Returns:
            输出特征图（如果使用shortcut则加上输入）
        """
        # 对4个方向分别进行padding和卷积，然后concat
        out = torch.cat([self.cv1(self.pad[g](x)) for g in range(4)], 1)
        out = self.cv2(out)
        # 如果使用残差连接则加上输入
        return x + out if self.add else out


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("APConv即插即用模块测试\n")
    
    # 创建测试输入张量 [B, C, H, W]
    x = torch.randn(2, 64, 32, 32)
    print(f"输入: {x.shape}")
    
    # 测试PConv
    pconv = PConv(c1=64, c2=128, k=3, s=1)
    out1 = pconv(x)
    print(f"PConv输出: {out1.shape}")
    
    # 测试APBottleneck
    ap_bottleneck = APBottleneck(c1=64, c2=64, shortcut=True)
    out2 = ap_bottleneck(x)
    print(f"APBottleneck输出: {out2.shape}")
    
    # 测试APC2f
    apc2f = APC2f(c1=64, c2=128, n=2, P=True)
    out3 = apc2f(x)
    print(f"APC2f输出: {out3.shape}")
    
    print("\n✓ 测试通过！模块可正常使用")
