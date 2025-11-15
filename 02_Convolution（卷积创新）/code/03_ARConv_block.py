"""ARConv 模块

该文件实现了自适应感受野卷积（Adaptive Receptive-field Convolution，ARConv）。
核心特性：
- 根据输入特征自动估计局部感受野尺寸（N_X, N_Y）
- 通过 `m_conv`、`b_conv` 计算调制权重和偏置
- 使用 `p_conv`、`l_conv`、`w_conv` 学习可变形偏移，实现空间自适应采样
- 在子模块上注册 backward hook，用于缩放梯度，避免训练不稳定

使用时仅需导入 `ARConv` 并调用 `forward(x, epoch, hw_range)`，
同时确保训练阶段传入递增的 `epoch`，推理时可手动调用 `remove_hooks()` 释放资源。
"""

import torch
import torch.nn as nn


class ARConv(nn.Module):
    """Adaptive receptive-field convolution with learnable offsets and gradient scaling.

    Args:
        inc: 输入通道数。
        outc: 输出通道数。
        kernel_size: 基础卷积核大小，默认 3。
        padding: 零填充大小，默认 1。
        stride: 步长，默认 1。
        l_max: 可搜索的最大感受野高度。
        w_max: 可搜索的最大感受野宽度。
        flag: 预留开关参数。
        modulation: 是否启用调制功能。

    Note:
        初始化过程中会在若干子层上注册 backward hook，将梯度缩放为原来的 0.1，
        以缓解训练震荡。如果实例仅用于一次性测试，记得调用 `remove_hooks()`。
    """

    def __init__(
        self,
        inc: int,
        outc: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 1,
        l_max: int = 9,
        w_max: int = 9,
        flag: bool = False,
        modulation: bool = True,
    ) -> None:
        super().__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    inc,
                    outc,
                    kernel_size=(i // 10, i % 10),
                    stride=(i // 10, i % 10),
                    padding=0,
                )
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh(),
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = [
            self.m_conv[0].register_full_backward_hook(self._set_lr),
            self.m_conv[1].register_full_backward_hook(self._set_lr),
            self.b_conv[0].register_full_backward_hook(self._set_lr),
            self.b_conv[1].register_full_backward_hook(self._set_lr),
            self.p_conv[0].register_full_backward_hook(self._set_lr),
            self.p_conv[1].register_full_backward_hook(self._set_lr),
            self.l_conv[0].register_full_backward_hook(self._set_lr),
            self.l_conv[1].register_full_backward_hook(self._set_lr),
            self.w_conv[0].register_full_backward_hook(self._set_lr),
            self.w_conv[1].register_full_backward_hook(self._set_lr),
        ]

        self.reserved_NXY = nn.Parameter(
            torch.tensor([3, 3], dtype=torch.int32),
            requires_grad=False,
        )

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self) -> None:
        """Remove all registered backward hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def forward(self, x, epoch, hw_range):
        assert (
            isinstance(hw_range, list) and len(hw_range) == 2
        ), "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)

            def phi(val):
                if val % 2 == 0:
                    val -= 1
                return val

            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)
            if epoch == 100:
                self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False,
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]

        N = N_X * N_Y
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
            1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
            1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        x_offset = (
            g_lt.unsqueeze(dim=1) * x_q_lt
            + g_rb.unsqueeze(dim=1) * x_q_rb
            + g_lb.unsqueeze(dim=1) * x_q_lb
            + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat(
            [
                x_offset[..., s : s + n_y].contiguous().view(b, c, h, w * n_y)
                for s in range(0, N, n_y)
            ],
            dim=-1,
        )
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset


def _demo_forward():
    """Run a sanity check forward pass."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = ARConv(inc=3, outc=8).to(device)
    dummy_input = torch.randn(2, 3, 32, 32, device=device)
    output = layer(dummy_input, epoch=10, hw_range=[1, 9])
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    torch.manual_seed(0)
    _demo_forward()

