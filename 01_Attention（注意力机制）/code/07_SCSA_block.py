import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SCSAPlug(nn.Module):
	"""
	Standalone, plug-and-play SCSA block (no external framework dependencies).

	Structure matches the paper figure at a high level:
	- Shared Multi-Semantic Spatial Attention (SMSA):
	  1D depthwise convs with multiple receptive fields along H and W.
	- Progressive Channel-wise Self-Attention (PCSA):
	  Downsample -> single/multi-head channel self-attention -> upsample.

	Input/Output: (B, C, H, W) -> (B, C, H, W)

	Args:
		dim: input channels C (must be divisible by 4 for SMSA grouping)
		head_num: number of channel attention heads (>=1)
		window_size: downsample window size used in PCSA
		group_kernel_sizes: four kernel sizes for SMSA [local, small, mid, large]
		down_sample_mode: 'avg' or 'max'
		gate: 'sigmoid' or 'tanh' for SMSA gate
	"""

	def __init__(
		self,
		dim: int,
		head_num: int = 1,
		window_size: int = 7,
		group_kernel_sizes: List[int] = [3, 5, 7, 9],
		down_sample_mode: str = "avg",
		gate: str = "sigmoid",
	):
		super().__init__()
		assert dim % 4 == 0, "dim must be divisible by 4 (for 4-way split in SMSA)."
		assert head_num >= 1 and dim % head_num == 0, "dim must be divisible by head_num."

		self.dim = dim
		self.head_num = head_num
		self.head_dim = dim // head_num
		self.scale = 1.0 / math.sqrt(self.head_dim)
		self.window_size = window_size
		self.down_sample_mode = down_sample_mode

		# --- SMSA (H/W 1D depthwise, multi-kernel, shared by channel groups) ---
		group_ch = dim // 4
		k1, k2, k3, k4 = group_kernel_sizes
		self.dw1_h = nn.Conv1d(group_ch, group_ch, k1, padding=k1 // 2, groups=group_ch)
		self.dw2_h = nn.Conv1d(group_ch, group_ch, k2, padding=k2 // 2, groups=group_ch)
		self.dw3_h = nn.Conv1d(group_ch, group_ch, k3, padding=k3 // 2, groups=group_ch)
		self.dw4_h = nn.Conv1d(group_ch, group_ch, k4, padding=k4 // 2, groups=group_ch)

		self.dw1_w = nn.Conv1d(group_ch, group_ch, k1, padding=k1 // 2, groups=group_ch)
		self.dw2_w = nn.Conv1d(group_ch, group_ch, k2, padding=k2 // 2, groups=group_ch)
		self.dw3_w = nn.Conv1d(group_ch, group_ch, k3, padding=k3 // 2, groups=group_ch)
		self.dw4_w = nn.Conv1d(group_ch, group_ch, k4, padding=k4 // 2, groups=group_ch)

		# GroupNorm with 4 groups as in the figure (n=4)
		self.gn_h = nn.GroupNorm(4, dim)
		self.gn_w = nn.GroupNorm(4, dim)
		self.gate = nn.Sigmoid() if gate == "sigmoid" else nn.Tanh()

		# --- PCSA (channel self-attention) ---
		if down_sample_mode == "avg":
			self.pool = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
		else:
			self.pool = nn.MaxPool2d(kernel_size=window_size, stride=window_size)

		# light projection before attention (optional but stabilizes)
		self.pre = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
		self.norm = nn.GroupNorm(1, dim)

		# q, k, v along channels (1x1 conv keeps spatial)
		self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
		self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
		self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
		self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

	def _spatial_multi_semantic_attn(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.shape
		group_ch = c // 4

		# (B, C, H) and (B, C, W) via GAP along the other axis
		x_h = x.mean(dim=3)  # (B, C, H)
		x_w = x.mean(dim=2)  # (B, C, W)

		lh, gh_s, gh_m, gh_l = torch.split(x_h, group_ch, dim=1)
		lw, gw_s, gw_m, gw_l = torch.split(x_w, group_ch, dim=1)

		# depthwise convs along H and W
		y_h = torch.cat([
			self.dw1_h(lh), self.dw2_h(gh_s), self.dw3_h(gh_m), self.dw4_h(gh_l)
		], dim=1)  # (B, C, H)
		y_w = torch.cat([
			self.dw1_w(lw), self.dw2_w(gw_s), self.dw3_w(gw_m), self.dw4_w(gw_l)
		], dim=1)  # (B, C, W)

		# group norm + gate
		# reshape to (B, C, H, 1) / (B, C, 1, W) so GroupNorm sees channels
		y_h_2d = y_h.view(b, c, h, 1)
		y_w_2d = y_w.view(b, c, 1, w)
		a_h = self.gate(self.gn_h(y_h_2d))
		a_w = self.gate(self.gn_w(y_w_2d))

		return x * a_h * a_w

	def _channel_self_attention(self, x: torch.Tensor) -> torch.Tensor:
		b, c, h, w = x.shape

		y = self.pool(x)  # (B, C, h', w')
		y = self.pre(y)
		y = self.norm(y)

		q = self.q(y)
		k = self.k(y)
		v = self.v(y)

		# reshape to (B, heads, head_dim, N)
		heads = self.head_num
		hdim = self.head_dim
		n_tokens = q.shape[2] * q.shape[3]

		def to_bhcn(t: torch.Tensor) -> torch.Tensor:
			return t.view(b, heads, hdim, n_tokens)

		qh = to_bhcn(q)
		kh = to_bhcn(k)
		vh = to_bhcn(v)

		attn = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # (B, heads, hdim, hdim)
		attn = torch.softmax(attn, dim=-1)
		out = torch.matmul(attn, vh)  # (B, heads, hdim, N)

		out = out.view(b, c, q.shape[2], q.shape[3])
		out = self.proj(out)
		# upsample back to original spatial size
		out = F.interpolate(out, size=(h, w), mode="nearest")
		return out

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		# 1) spatial attention
		x_spa = self._spatial_multi_semantic_attn(x)
		# 2) channel self-attention (progressive)
		x_cha = self._channel_self_attention(x_spa)
		# 3) residual refinement
		return x_spa + x_cha


def _demo_input(bs: int, c: int, h: int, w: int, device: torch.device) -> torch.Tensor:
	return torch.randn(bs, c, h, w, device=device)


def main():
	# Minimal manual args to keep the file simple
	bs, c, h, w = 2, 64, 56, 56
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = SCSAPlug(dim=c, head_num=1, window_size=7).to(device).eval()
	x = _demo_input(bs, c, h, w, device)
	with torch.inference_mode():
		y = model(x)

	print("SCSAPlug Demo")
	print(f"Input : {tuple(x.shape)} on {device}")
	print(f"Output: {tuple(y.shape)}")


if __name__ == "__main__":
	main()


