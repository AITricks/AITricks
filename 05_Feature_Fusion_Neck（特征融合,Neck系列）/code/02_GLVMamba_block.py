import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except Exception:  # pragma: no cover
    selective_scan_fn, selective_scan_ref = None, None


def _get_selective_scan_impl():
    if selective_scan_fn is not None:
        return selective_scan_fn
    if selective_scan_ref is not None:
        return selective_scan_ref

    raise ImportError(
        "Neither selective_scan_fn nor selective_scan_ref could be imported from "
        "`mamba_ssm`. Please install `mamba_ssm` to use the SS2D module."
    )


class SELayer(nn.Module):
    """Squeeze-and-Excitation block shown in Fig.5."""

    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class h_sigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class ECALayer(nn.Module):
    """Efficient Channel Attention helper used in LocalityFeedForward."""

    def __init__(self, channel: int, gamma: int = 2, b: int = 1, sigmoid: bool = True):
        super().__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid() if sigmoid else h_sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LocalityFeedForward(nn.Module):
    """Locality FeedForward block highlighted in the GLSS2D structure (Fig.4)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        stride: int,
        expand_ratio: float = 4.0,
        act: str = "hs+se",
        reduction: int = 4,
        wo_dp_conv: bool = False,
        dp_first: bool = False,
    ):
        super().__init__()
        hidden_dim = int(in_dim * expand_ratio)
        kernel_size = 3

        layers = [
            nn.Conv2d(in_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=True),
        ]

        if not wo_dp_conv:
            dp = [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if act.find("hs") >= 0 else nn.ReLU6(inplace=True),
            ]
            layers = dp + layers if dp_first else layers + dp

        if act.find("+") >= 0:
            attn = act.split("+")[1]
            if attn == "se":
                layers.append(SELayer(hidden_dim, reduction=reduction))
            elif attn.find("eca") >= 0:
                layers.append(ECALayer(hidden_dim, sigmoid=attn == "eca"))
            else:
                raise NotImplementedError(f"Activation type {act} is not implemented")

        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_dim),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SS2D(nn.Module):
    """2D Selective Scanning block (Fig.3)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 0.5,
        dt_rank: Union[str, int] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = tuple(
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = tuple(
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(4)
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        selective_scan = _get_selective_scan_impl()
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)

        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y).to(x.dtype)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_corev0(x)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    """Original VSS block (Fig.4a) built on top of SS2D."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_dim)
        self.proj_in = nn.Linear(hidden_dim, hidden_dim)
        self.attn = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        self.ffn = FeedForward(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.proj_in(self.norm1(x))))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class GLSS2D(nn.Module):
    """Global-local selective scanning block (Fig.4c)."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("GLSS2D expects an even feature dimension so it can split channels in half.")
        self.global_dim = dim // 2
        self.global_branch = SS2D(d_model=self.global_dim, d_state=d_state, dropout=dropout)
        self.local_branch = LocalityFeedForward(
            in_dim=self.global_dim, out_dim=self.global_dim, stride=1, act="hs+se"
        )
        self.fuse_norm = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_global, x_local = torch.chunk(x, 2, dim=-1)
        global_feat = self.global_branch(x_global)
        local_feat = self.local_branch(x_local.permute(0, 3, 1, 2).contiguous())
        local_feat = local_feat.permute(0, 2, 3, 1).contiguous()
        fused = torch.cat([global_feat, local_feat], dim=-1)
        fused = fused + self.feed_forward(self.fuse_norm(fused))
        return fused


class GLVSSBlock(nn.Module):
    """Global-Local VSS block (Fig.4b) that wraps GLSS2D."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int = 16,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(hidden_dim)
        self.pre_linear = nn.Linear(hidden_dim, hidden_dim)
        self.glss2d = GLSS2D(hidden_dim, d_state=d_state, mlp_ratio=mlp_ratio, dropout=attn_drop_rate)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(hidden_dim)
        self.ffn = FeedForward(hidden_dim, int(hidden_dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.glss2d(self.pre_linear(self.norm1(x))))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


def main():
    """Light-weight sanity checks for the plug-and-play modules."""
    torch.manual_seed(0)

    se = SELayer(channel=32, reduction=8)
    se_out = se(torch.randn(2, 32, 16, 16))
    print("SELayer output:", tuple(se_out.shape))

    lff = LocalityFeedForward(in_dim=32, out_dim=32, stride=1, act="hs+se")
    lff_out = lff(torch.randn(2, 32, 16, 16))
    print("LocalityFeedForward output:", tuple(lff_out.shape))

    try:
        ss2d = SS2D(d_model=32, d_state=8)
        ss2d_out = ss2d(torch.randn(2, 8, 8, 32))
        print("SS2D output:", tuple(ss2d_out.shape))
    except ImportError as exc:
        print(f"Skipping SS2D test because {exc}")

    feat = torch.randn(2, 8, 8, 64)
    try:
        vss = VSSBlock(hidden_dim=64, d_state=8)
        print("VSSBlock output:", tuple(vss(feat).shape))

        glss = GLSS2D(dim=64, d_state=8)
        print("GLSS2D output:", tuple(glss(feat).shape))

        glvss = GLVSSBlock(hidden_dim=64, d_state=8)
        print("GLVSSBlock output:", tuple(glvss(feat).shape))
    except ImportError as exc:
        print(f"Skipping VSS/GLSS tests because {exc}")


if __name__ == "__main__":
    main()

