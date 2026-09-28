"""
Causal/online target-speaker-extraction (TSE) model.

Conditioning paths
------------------
1. A fixed reference/enrollment waveform is encoded into compressed complex
   STFT memory. Each mixture frame attends only to this fixed reference memory.
2. The aligned mixture/reference TF features enter a strictly causal local-global
   conditioner:
      - local branch: causal MuGI interaction for short-range TF details;
      - global branch: bidirectional frequency modeling at each current frame and
        unidirectional Mamba modeling over mixture time;
      - asymmetric gate: the reference modulates a mixture-centered base feature.
3. A frozen ECAPA speaker embedding from the same reference is injected into
   every separator block with weak, zero-initialized FiLM conditioning.

Interface
---------
    forward(x, aux, reference_lengths=None)

    x:                 mixture waveform,   [B, T] or [B, 1, T]
    aux:               padded reference waveform, [B, R] or [B, 1, R]
    reference_lengths: optional true reference lengths [B]. Values may be
                       absolute sample counts or relative lengths in (0, 1].

When reference_lengths is provided, each reference is cropped to its true
length and encoded independently inside the batch. Padding therefore never
enters spectral reference conditioning or the frozen speaker encoder.

Output
------
    extracted target speech: [B, 1, T]

The mixture-time backbone is causal. The reference is assumed to be available
before online extraction starts and can be cached once for streaming inference.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from speechbrain.inference import EncoderClassifier
except Exception:
    EncoderClassifier = None


# -----------------------------------------------------------------------------
# Optional fused selective-scan kernel.
# A pure-PyTorch fallback is provided so the file remains runnable without the
# compiled mamba_ssm extension. The fallback is much slower and is intended for
# correctness checks rather than deployment.
# -----------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as _fused_selective_scan_fn

    HAS_FUSED_SELECTIVE_SCAN = True
except Exception:
    _fused_selective_scan_fn = None
    HAS_FUSED_SELECTIVE_SCAN = False


_WARNED_ABOUT_SCAN_FALLBACK = False


def _reference_selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
):
    """Slow PyTorch implementation matching the selective-scan tensor contract.

    Args:
        u, delta: [B, D_inner, L]
        A:        [D_inner, D_state]
        B, C:     [B, D_state, L]
        D:        [D_inner]
        z:        [B, D_inner, L]
    """
    global _WARNED_ABOUT_SCAN_FALLBACK
    if not _WARNED_ABOUT_SCAN_FALLBACK:
        warnings.warn(
            "mamba_ssm fused selective_scan_fn was not found. Falling back to "
            "a slow PyTorch implementation. Install mamba_ssm for training and "
            "real-time deployment.",
            RuntimeWarning,
        )
        _WARNED_ABOUT_SCAN_FALLBACK = True

    if delta_bias is not None:
        delta = delta + delta_bias.view(1, -1, 1)
    if delta_softplus:
        delta = F.softplus(delta)

    batch, inner_dim, seq_len = u.shape
    state_dim = A.shape[-1]
    state = torch.zeros(
        batch,
        inner_dim,
        state_dim,
        device=u.device,
        dtype=torch.float32,
    )

    u_float = u.float()
    delta_float = delta.float()
    A_float = A.float()
    B_float = B.float()
    C_float = C.float()
    D_float = D.float()

    outputs: List[torch.Tensor] = []
    for t in range(seq_len):
        dt = delta_float[:, :, t]  # [B, D]
        u_t = u_float[:, :, t]     # [B, D]
        B_t = B_float[:, :, t]     # [B, N]
        C_t = C_float[:, :, t]     # [B, N]

        delta_A = torch.exp(dt.unsqueeze(-1) * A_float.unsqueeze(0))
        delta_B_u = dt.unsqueeze(-1) * B_t.unsqueeze(1) * u_t.unsqueeze(-1)
        state = delta_A * state + delta_B_u

        y_t = (state * C_t.unsqueeze(1)).sum(dim=-1)
        y_t = y_t + D_float.unsqueeze(0) * u_t
        if z is not None:
            y_t = y_t * F.silu(z[:, :, t].float())
        outputs.append(y_t.to(dtype=u.dtype))

    y = torch.stack(outputs, dim=-1)
    if return_last_state:
        return y, state
    return y


def selective_scan(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor] = None,
    delta_bias: Optional[torch.Tensor] = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
):
    if HAS_FUSED_SELECTIVE_SCAN:
        return _fused_selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=delta_softplus,
            return_last_state=return_last_state,
        )
    return _reference_selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=return_last_state,
    )


# -----------------------------------------------------------------------------
# Small self-contained utility layers. These replace the timm dependency.
# -----------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        return self.drop2(x)


class Attention(nn.Module):
    """Self-attention used only along frequency, never along streaming time."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch, length, channels)
        return self.proj_drop(self.proj(x))


# -----------------------------------------------------------------------------
# Causal analysis/synthesis frontend.
# torch.stft(center=True) is not used because it reads future samples.
# -----------------------------------------------------------------------------
@dataclass
class STFTMeta:
    original_length: int
    left_pad: int
    padded_length: int


class CausalSTFTEncoder(nn.Module):
    def __init__(self, win_length: int = 512, hop_length: int = 256, n_fft: int = 512):
        super().__init__()
        if n_fft < win_length:
            raise ValueError("n_fft must be >= win_length")
        if not 0 < hop_length <= win_length:
            raise ValueError("hop_length must satisfy 0 < hop_length <= win_length")
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.left_pad = self.win_length - self.hop_length

        # End-of-stream zero-padding guard. This does not use future speech;
        # it keeps the final real samples inside a stable WOLA overlap region.
        self.tail_pad = self.win_length - self.hop_length

        self.register_buffer(
            "window",
            torch.sqrt(torch.hann_window(self.win_length).clamp_min(0.0)),
            persistent=False,
        )

    @staticmethod
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            if x.shape[1] != 1:
                raise ValueError(f"Expected mono input [B,1,T], got {tuple(x.shape)}")
            x = x[:, 0]
        if x.ndim != 2:
            raise ValueError(f"Expected [T], [B,T], or [B,1,T], got {tuple(x.shape)}")
        return x

    def num_frames(self, input_length: int) -> int:
        padded_length = (
            int(input_length)
            + self.left_pad
            + self.tail_pad
        )
        if padded_length < self.win_length:
            padded_length = self.win_length
        else:
            remainder = (padded_length - self.win_length) % self.hop_length
            padded_length += (self.hop_length - remainder) % self.hop_length
        return 1 + (padded_length - self.win_length) // self.hop_length

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, STFTMeta]:
        # FFT and complex operations are kept in FP32 under BF16 mixed precision.
        x = self._to_2d(x).float()
        original_length = x.shape[-1]

        # Left history padding + right end-of-stream protection padding.
        x = F.pad(x, (self.left_pad, self.tail_pad))

        if x.shape[-1] < self.win_length:
            alignment_pad = self.win_length - x.shape[-1]
        else:
            remainder = (x.shape[-1] - self.win_length) % self.hop_length
            alignment_pad = (self.hop_length - remainder) % self.hop_length

        if alignment_pad > 0:
            x = F.pad(x, (0, alignment_pad))

        frames = x.unfold(-1, self.win_length, self.hop_length)  # [B,Tf,W]
        frames = frames * self.window.to(device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        spectrum_ri = torch.stack([spectrum.real, spectrum.imag], dim=1)

        return spectrum_ri, STFTMeta(
            original_length=original_length,
            left_pad=self.left_pad,
            padded_length=x.shape[-1],
        )


class CausalISTFTDecoder(nn.Module):
    def __init__(self, win_length: int = 512, hop_length: int = 256, n_fft: int = 512):
        super().__init__()
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.register_buffer(
            "window",
            torch.sqrt(torch.hann_window(self.win_length).clamp_min(0.0)),
            persistent=False,
        )

    def forward(self, spectrum_ri: torch.Tensor, meta: STFTMeta) -> torch.Tensor:
        if spectrum_ri.ndim != 5 or spectrum_ri.shape[2] != 2:
            raise ValueError(
                "Expected enhanced spectrum [B,S,2,T,F], got "
                f"{tuple(spectrum_ri.shape)}"
            )

        batch, sources, _, frames_count, _ = spectrum_ri.shape

        # torch.complex and iFFT do not accept BF16 reliably.
        spectrum_ri = spectrum_ri.float()
        spectrum = torch.complex(spectrum_ri[:, :, 0], spectrum_ri[:, :, 1])
        frames = torch.fft.irfft(spectrum, n=self.n_fft, dim=-1)[..., : self.win_length]
        window = self.window.to(device=frames.device, dtype=frames.dtype)
        frames = frames * window

        flat_frames = frames.reshape(batch * sources, frames_count, self.win_length)
        flat_frames = flat_frames.transpose(1, 2).contiguous()  # [B*S,W,Tf]
        output_length = (frames_count - 1) * self.hop_length + self.win_length

        waveform = F.fold(
            flat_frames,
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).view(batch * sources, output_length)

        window_sq = window.square().view(1, self.win_length, 1)
        window_sq = window_sq.expand(batch * sources, -1, frames_count)
        denominator = F.fold(
            window_sq,
            output_size=(1, output_length),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).view(batch * sources, output_length)
        # Never amplify network-predicted samples by dividing through a
        # near-zero Hann overlap denominator.
        denominator_floor = 1.0e-4
        valid = denominator > denominator_floor
        waveform = torch.where(
            valid,
            waveform / denominator.clamp_min(denominator_floor),
            torch.zeros_like(waveform),
        )

        start = meta.left_pad
        end = start + meta.original_length
        waveform = waveform[:, start:end]
        waveform = waveform.view(batch, sources, meta.original_length)
        return torch.nan_to_num(waveform)


# -----------------------------------------------------------------------------
# Causal TF feature extraction.
# -----------------------------------------------------------------------------
class CausalConv2dTF(nn.Module):
    """2-D convolution causal on T and symmetric on F for [B,C,T,F]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        dilation: Tuple[int, int] = (1, 1),
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        kt, kf = kernel_size
        dt, df = dilation
        self.left_pad_t = dt * (kt - 1)
        total_pad_f = df * (kf - 1)
        self.left_pad_f = total_pad_f // 2
        self.right_pad_f = total_pad_f - self.left_pad_f
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.left_pad_f, self.right_pad_f, self.left_pad_t, 0),
        )
        return self.conv(x)


class CausalMultiRangeEmbedding(nn.Module):
    def __init__(self, in_channels: int = 2, branch_dim: int = 96, bias: bool = False):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                CausalConv2dTF(in_channels, branch_dim, (1, 1), (1, 1), bias=bias),
                CausalConv2dTF(in_channels, branch_dim, (3, 3), (1, 1), bias=bias),
                CausalConv2dTF(in_channels, branch_dim, (3, 3), (2, 2), bias=bias),
                CausalConv2dTF(in_channels, branch_dim, (3, 3), (3, 3), bias=bias),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class TimeIndependentGroupNorm(nn.Module):
    """GroupNorm independently for every time frame; no future-time leakage."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, freqs = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch * frames, channels, freqs)
        x = self.norm(x)
        return x.view(batch, frames, channels, freqs).permute(0, 2, 1, 3).contiguous()


# -----------------------------------------------------------------------------
# Mamba/attention blocks.
# -----------------------------------------------------------------------------
class UniMambaTemporalMixer(nn.Module):
    """Unidirectional Mamba mixer for [B,L,C]."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        dt_rank: str | int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.expand * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
        )
        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + 2 * self.d_state,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.unsqueeze(0).repeat(self.d_inner, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2).contiguous()
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :seq_len])

        A = -torch.exp(self.A_log.float())
        x_flat = x.transpose(1, 2).reshape(batch * seq_len, self.d_inner)
        x_dbl = self.x_proj(x_flat)
        dt, B_param, C_param = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        # Apply only the projection weight here. The bias is supplied once to
        # selective_scan as delta_bias, matching the fused Mamba implementation.
        dt = F.linear(dt, self.dt_proj.weight, bias=None)
        dt = dt.view(batch, seq_len, self.d_inner).transpose(1, 2)
        B_param = B_param.view(batch, seq_len, self.d_state).transpose(1, 2).contiguous()
        C_param = C_param.view(batch, seq_len, self.d_state).transpose(1, 2).contiguous()

        y = selective_scan(
            x,
            dt,
            A,
            B_param,
            C_param,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = y.transpose(1, 2).contiguous()
        return self.out_proj(y)


class BiMambaFrequencyMixer(nn.Module):
    """Bidirectional only over F, which does not violate time causality."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.forward_mamba = UniMambaTemporalMixer(dim, d_state, d_conv, expand)
        self.backward_mamba = UniMambaTemporalMixer(dim, d_state, d_conv, expand)
        self.merge = nn.Linear(2 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_forward = self.forward_mamba(x)
        y_backward = torch.flip(
            self.backward_mamba(torch.flip(x, dims=[1])),
            dims=[1],
        )
        return self.merge(torch.cat([y_forward, y_backward], dim=-1))


class MambaMLPUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        bidirectional: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.norm_mamba = nn.LayerNorm(dim)
        if bidirectional:
            self.mamba = BiMambaFrequencyMixer(dim, d_state, d_conv, expand)
        else:
            self.mamba = UniMambaTemporalMixer(dim, d_state, d_conv, expand)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if layer_scale is None:
            self.gamma_mamba = 1.0
            self.gamma_mlp = 1.0
        else:
            self.gamma_mamba = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma_mlp = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_mamba * self.mamba(self.norm_mamba(x)))
        x = x + self.drop_path(self.gamma_mlp * self.mlp(self.norm_mlp(x)))
        return x


class AttentionMLPUnit(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale: Optional[float] = 1e-5,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, proj_drop=drop)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dim, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        if layer_scale is None:
            self.gamma_attn = 1.0
            self.gamma_mlp = 1.0
        else:
            self.gamma_attn = nn.Parameter(layer_scale * torch.ones(dim))
            self.gamma_mlp = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.gamma_attn * self.attn(self.norm_attn(x)))
        x = x + self.drop_path(self.gamma_mlp * self.mlp(self.norm_mlp(x)))
        return x


class FrequencyModelingBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        layer_scale: float = 1e-5,
    ):
        super().__init__()
        self.mamba = MambaMLPUnit(
            dim,
            mlp_ratio,
            d_state,
            d_conv,
            expand,
            bidirectional=True,
            layer_scale=layer_scale,
        )
        self.attention = AttentionMLPUnit(
            dim,
            num_heads,
            mlp_ratio,
            layer_scale=layer_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self.mamba(x))


class OnlineFreqTimeBlock(nn.Module):
    """Full-frequency modeling plus strictly causal temporal modeling."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        num_time_units: int = 2,
        layer_scale: float = 1e-5,
    ):
        super().__init__()
        self.channels = channels
        self.local_causal_conv = nn.Sequential(
            CausalConv2dTF(channels, channels, (3, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            TimeIndependentGroupNorm(1, channels),
            nn.PReLU(channels),
        )
        self.freq_block = FrequencyModelingBlock(
            channels,
            num_heads,
            mlp_ratio,
            d_state,
            d_conv,
            expand,
            layer_scale,
        )
        self.time_units = nn.ModuleList(
            [
                MambaMLPUnit(
                    channels,
                    mlp_ratio,
                    d_state,
                    d_conv,
                    expand,
                    bidirectional=False,
                    layer_scale=layer_scale,
                )
                for _ in range(num_time_units)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, freqs = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        residual = x
        x = x + self.local_causal_conv(x)

        # Frequency sequence at each fixed time frame: [B*T,F,C].
        xf = x.permute(0, 2, 3, 1).contiguous().view(batch * frames, freqs, channels)
        xf = self.freq_block(xf)
        x = xf.view(batch, frames, freqs, channels).permute(0, 3, 1, 2).contiguous()

        # Strictly causal time sequence at each frequency: [B*F,T,C].
        xt = x.permute(0, 3, 2, 1).contiguous().view(batch * freqs, frames, channels)
        for unit in self.time_units:
            xt = unit(xt)
        x = xt.view(batch, freqs, frames, channels).permute(0, 3, 2, 1).contiguous()
        return x + residual


# -----------------------------------------------------------------------------
# Reference-conditioning blocks.
# -----------------------------------------------------------------------------
class TimeIndependentLayerNorm2d(nn.Module):
    """LayerNorm over channels independently at every (time, frequency) bin."""

    def __init__(self, channels: int, eps: float = 1.0e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,F] -> [B,T,F,C] -> LN(C) -> [B,C,T,F]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(
            x,
            normalized_shape=(x.shape[-1],),
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )
        return x.permute(0, 3, 1, 2).contiguous()


class DualStreamGate(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=1)
        y1, y2 = y.chunk(2, dim=1)
        return x1 * y2, y1 * x2


class DualStreamSeq(nn.Sequential):
    def forward(
        self,
        x: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        y = x if y is None else y
        for module in self:
            x, y = module(x, y)
        return x, y


class DualStreamBlock(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.seq = nn.Sequential()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.seq.add_module(key, module)
        else:
            for index, module in enumerate(args):
                self.seq.add_module(str(index), module)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.seq(x), self.seq(y)


class CausalChannelGate(nn.Module):
    """Local 1x1 channel gate; it does not pool over future time frames."""

    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.proj(x))


class CausalMuGIBlock(nn.Module):
    """Causal mixture/reference dual-stream interaction block."""

    def __init__(self, channels: int, shared_b: bool = False):
        super().__init__()
        self.block1 = DualStreamSeq(
            DualStreamBlock(
                TimeIndependentLayerNorm2d(channels),
                nn.Conv2d(channels, channels * 2, kernel_size=1),
                CausalConv2dTF(
                    channels * 2,
                    channels * 2,
                    kernel_size=(3, 3),
                    groups=channels * 2,
                ),
            ),
            DualStreamGate(),
            DualStreamBlock(CausalChannelGate(channels)),
            DualStreamBlock(nn.Conv2d(channels, channels, kernel_size=1)),
        )

        self.a_left = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.a_right = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.block2 = DualStreamSeq(
            DualStreamBlock(
                TimeIndependentLayerNorm2d(channels),
                nn.Conv2d(channels, channels * 2, kernel_size=1),
            ),
            DualStreamGate(),
            DualStreamBlock(nn.Conv2d(channels, channels, kernel_size=1)),
        )

        self.shared_b = bool(shared_b)
        if self.shared_b:
            self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.b_left = nn.Parameter(torch.zeros(1, channels, 1, 1))
            self.b_right = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left, right = self.block1(mixture, reference)
        left_skip = mixture + self.a_left * left
        right_skip = reference + self.a_right * right

        left, right = self.block2(left_skip, right_skip)
        if self.shared_b:
            return left_skip + self.b * left, right_skip + self.b * right
        return (
            left_skip + self.b_left * left,
            right_skip + self.b_right * right,
        )


class CausalLocalGlobalReferenceConditioner(nn.Module):   # 包含了基础分支，本地分支，全局分支，门控融合
    """Strictly causal local-global interaction between mixture and reference.

    Inputs:
        mixture:   [B, C_interaction, T, F]  # mixtrue是原始的mixture
        reference: [B, C_interaction, T, F]  # reference是带有相似度的和mixture的形状相似

    Output:
        conditioned mixture feature: [B, C_out, T, F]

    Causality guarantee along mixture time T
    -----------------------------------------
    * Local branch: CausalMuGIBlock uses only 1x1 operations and
      CausalConv2dTF, whose temporal padding is left-only.
    * Global branch: frequency modeling is bidirectional only inside one fixed
      time frame. Temporal modeling uses unidirectional Mamba.
    * Gate: CausalConv2dTF uses only the current and previous mixture frames.
    * Normalization is time-independent and never aggregates future frames.

    The reference feature has already been aligned to mixture time by the
    fixed-reference cross-attention in compute_similarity_batch(). Therefore,
    both inputs have the same T, but the reference memory itself is available
    before streaming begins.
    """

    VALID_MODES = {"local", "global", "local_global"}   ##### 方便做消融实验，设计了local配置，global配置，以及local_global配置

    def __init__(
        self,
        interaction_channels: int,
        output_channels: int,
        num_local_blocks: int = 2,
        num_global_blocks: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        global_num_time_units: int = 1,
        layer_scale: float = 1.0e-5,
        eps: float = 1.0e-5,
        mode: str = "local_global",
        fusion_init_scale: float = 1.0,
    ):
        super().__init__()

        if interaction_channels <= 0 or output_channels <= 0:
            raise ValueError("interaction_channels and output_channels must be positive")
        if num_local_blocks < 0 or num_global_blocks < 0:
            raise ValueError("branch block counts must be non-negative")
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"mode must be one of {sorted(self.VALID_MODES)}, got {mode!r}"
            )
        if mode in {"local", "local_global"} and num_local_blocks == 0:
            raise ValueError("the selected mode requires num_local_blocks > 0")
        if mode in {"global", "local_global"} and num_global_blocks == 0:
            raise ValueError("the selected mode requires num_global_blocks > 0")
        if output_channels % num_heads != 0:
            raise ValueError(
                f"output_channels={output_channels} must be divisible by "
                f"num_heads={num_heads}"
            )
        if fusion_init_scale < 0:
            raise ValueError("fusion_init_scale must be non-negative")

        self.interaction_channels = int(interaction_channels)
        self.output_channels = int(output_channels)
        self.mode = str(mode)
        self.use_local = mode in {"local", "local_global"}
        self.use_global = mode in {"global", "local_global"}

        # ------------------------------------------------------------------
        # Mixture-centered residual base.
        # ------------------------------------------------------------------
        # The extracted signal must remain anchored to the mixture. Reference
        # features are therefore used to predict a residual conditioning term,
        # rather than being treated as an equal signal source.
        self.base_projection = nn.Sequential(
            TimeIndependentLayerNorm2d(interaction_channels, eps=eps),
            nn.Conv2d(
                interaction_channels,
                output_channels,
                kernel_size=1,
                bias=False,
            ),
            TimeIndependentGroupNorm(1, output_channels, eps=eps),
            nn.PReLU(output_channels),
        )

        # ------------------------------------------------------------------
        # Local branch: the original causal MuGI interaction.
        # ------------------------------------------------------------------
        if self.use_local:
            self.local_interaction = DualStreamSeq(
                *[
                    CausalMuGIBlock(interaction_channels)
                    for _ in range(num_local_blocks)
                ]
            )
            self.local_projection = nn.Sequential(
                CausalConv2dTF(
                    interaction_channels * 2,
                    output_channels,
                    kernel_size=(3, 3),
                    bias=False,
                ),
                TimeIndependentGroupNorm(1, output_channels, eps=eps),
                nn.PReLU(output_channels),
            )
        else:
            self.local_interaction = None
            self.local_projection = None

        # ------------------------------------------------------------------
        # Global branch.
        # ------------------------------------------------------------------
        # First project each stream to output_channels. The projection is
        # shared, so mixture and reference remain in one comparable space.
        if self.use_global:
            self.global_stream_projection = nn.Sequential(
                TimeIndependentLayerNorm2d(interaction_channels, eps=eps),
                nn.Conv2d(
                    interaction_channels,
                    output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                TimeIndependentGroupNorm(1, output_channels, eps=eps),
                nn.PReLU(output_channels),
            )

            # Four complementary interaction descriptors are used:
            #   m:          mixture representation;
            #   r:          reference-conditioned representation;
            #   m * r:      signed feature agreement/correlation;
            #   |m - r|:    mismatch magnitude without an unstable direction sign.
            # The 1x1 projection is pointwise in time and therefore causal.
            self.global_joint_projection = nn.Sequential(
                nn.Conv2d(
                    output_channels * 4,
                    output_channels,
                    kernel_size=1,
                    bias=False,
                ),
                TimeIndependentGroupNorm(1, output_channels, eps=eps),
                nn.PReLU(output_channels),
            )

            # OnlineFreqTimeBlock is safe here:
            #   1) bidirectional operations run only along frequency F;
            #   2) temporal Mamba runs only from past to present.
            self.global_blocks = nn.ModuleList(
                [
                    OnlineFreqTimeBlock(
                        channels=output_channels,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                        num_time_units=global_num_time_units,
                        layer_scale=layer_scale,
                    )
                    for _ in range(num_global_blocks)
                ]
            )
        else:
            self.global_stream_projection = None
            self.global_joint_projection = None
            self.global_blocks = nn.ModuleList()

        # ------------------------------------------------------------------
        # Causal asymmetric gated fusion.
        # ------------------------------------------------------------------
        if self.use_local and self.use_global:
            self.gate = CausalConv2dTF(
                output_channels * 3,
                output_channels,
                kernel_size=(3, 3),
                bias=True,
            )
            # Zero logits -> sigmoid(0)=0.5, so training starts without an
            # arbitrary preference for either local or global information.
            nn.init.zeros_(self.gate.conv.weight)
            if self.gate.conv.bias is not None:
                nn.init.zeros_(self.gate.conv.bias)
        else:
            self.gate = None

        self.delta_projection = nn.Conv2d(
            output_channels,
            output_channels,
            kernel_size=1,
            bias=False,
        )

        # A very small channel-wise residual scale keeps the new conditioner
        # close to the mixture-centered base at initialization, while unlike an
        # exact zero it still lets both branches receive gradients immediately.
        self.fusion_gain = nn.Parameter(
            torch.full(
                (1, output_channels, 1, 1),
                float(fusion_init_scale),
            )
        )

    def _local_forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.local_interaction is None or self.local_projection is None:
            raise RuntimeError("local branch is disabled")
        local_mixture, local_reference = self.local_interaction(
            mixture,
            reference,
        )
        return self.local_projection(
            torch.cat([local_mixture, local_reference], dim=1)
        )

    def _global_forward(
        self,
        mixture: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.global_stream_projection is None
            or self.global_joint_projection is None
        ):
            raise RuntimeError("global branch is disabled")

        # The same projection is applied independently at every time frame.
        mixture_global = self.global_stream_projection(mixture)
        reference_global = self.global_stream_projection(reference)

        joint = torch.cat(  # Global branch 显式构造了 4 种信息
            [
                mixture_global,  # mixture的全局信息
                reference_global,  # reference的全局信息
                mixture_global * reference_global,  # mixture和reference的全局信息的点积（两者哪里相似/相关）
                torch.abs(mixture_global - reference_global),  # 差（模拟两者的不同）
            ],
            dim=1,
        )
        global_feature = self.global_joint_projection(joint)  # 然后压回全局信息
        for block in self.global_blocks:
            global_feature = block(global_feature)
        return global_feature

    def forward(
        self,
        mixture: torch.Tensor,   # 这里的输入已经是处理过的输入了，
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if mixture.shape != reference.shape:
            raise ValueError(
                "mixture/reference feature shapes must match, got "
                f"{tuple(mixture.shape)} and {tuple(reference.shape)}"
            )
        if mixture.ndim != 4:
            raise ValueError(
                f"expected [B,C,T,F] features, got {tuple(mixture.shape)}"
            )
        if mixture.shape[1] != self.interaction_channels:
            raise ValueError(
                f"expected {self.interaction_channels} interaction channels, "
                f"got {mixture.shape[1]}"
            )

        base = self.base_projection(mixture)  # 输入： [B, 256, T, F]， 输出：[B, 64, T, F]
        # 先让模型保留 mixture 的主体内容，reference 只负责告诉模型“从里面选谁”。

        local_feature: Optional[torch.Tensor] = None  # 然后是localbranch，采用的是MuGI
        # 因此 reference 会直接控制 mixture 的部分 feature。所以local branch 会更偏向短时间以及局部频率以及局部TF对应关系
        global_feature: Optional[torch.Tensor] = None
        if self.use_local:
            local_feature = self._local_forward(mixture, reference)
        if self.use_global:
            global_feature = self._global_forward(mixture, reference)

        if local_feature is not None and global_feature is not None:
            if self.gate is None:
                raise RuntimeError("gate was not initialized")
            gate = torch.sigmoid(    # ##########门控
                self.gate(    # #################这里的gate是一个卷积，但是进行0初始化了，所以是0
                    torch.cat(
                        [base, local_feature, global_feature],
                        dim=1,
                    )
                )
            )
            interaction = (  # interaction=0.5Local+0.5Global
                gate * local_feature
                + (1.0 - gate) * global_feature
            )
        elif local_feature is not None:
            interaction = local_feature
        elif global_feature is not None:
            interaction = global_feature
        else:
            raise RuntimeError("both reference-conditioning branches are disabled")

        delta = self.delta_projection(interaction)
        return base + self.fusion_gain * delta


@dataclass
class ReferenceConditioning:
    """Reference cache used by the separator.  # 参考缓存

    ``compressed_spectrum`` is the original fixed-length/batched representation
    used by legacy inference and streaming calls.

    ``compressed_spectrum_list`` is used when a training batch contains
    variable-length enrollment utterances. Each entry has shape
    [1, 2, T_ref_i, F] and contains no batch padding.
    """

    compressed_spectrum: Optional[torch.Tensor] = None
    speaker_embedding: Optional[torch.Tensor] = None
    compressed_spectrum_list: Optional[List[torch.Tensor]] = None

class OnlineTargetSpeakerExtractionModel(nn.Module):
    """
    Causal target-speaker extractor with one mixture input and one reference.

    Public training/evaluation interface:
        source = model(x, aux, reference_lengths=None)

    For variable-length batched enrollment, pass the true lengths so each
    reference is cropped and encoded independently before batch fusion.

    For streaming, encode_reference(aux) is called once and the returned cache is
    reused for every mixture chunk.
    """

    def __init__(
        self,
        dim: int = 64,
        mlp_ratio: float = 2.0,
        win_length: int = 512,
        hop_length: int = 256,
        n_fft: int = 512,
        num_layers: int = 6,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        num_time_units: int = 2,
        num_sources: int = 1,
        compression_factor: float = 0.5,
        mask_scale: float = 1.0,
        num_interaction_blocks: int = 2,
        num_global_interaction_blocks: int = 1,
        global_num_time_units: int = 1,
        reference_conditioning_mode: str = "local_global",
        interaction_fusion_init: float = 1.0,
        film_scale: float = 0.5,
        layer_scale: float = 1.0e-5,
        eps: float = 1.0e-5,
        lookahead_frames: int = 0,
        use_speaker_encoder: bool = True,
        speaker_encoder_source: str = (
            "/home/xueke/real_tse_challenge/spkrec-ecapa-voxceleb"
        ),
        speaker_embedding_dim: int = 192,
        speaker_encoder_device: Optional[str] = None,
    ):
        super().__init__()

        if num_sources != 1:
            raise ValueError("Target speaker extraction expects num_sources=1")
        if lookahead_frames != 0:
            raise ValueError(
                "This implementation is strictly causal over mixture time; "
                "lookahead_frames must be 0."
            )
        if mask_scale <= 0:
            raise ValueError(f"mask_scale must be positive, got {mask_scale}")
        if not 0.0 <= film_scale <= 1.0:
            raise ValueError(f"film_scale must be in [0,1], got {film_scale}")
        if num_interaction_blocks < 0:
            raise ValueError("num_interaction_blocks must be non-negative")
        if num_global_interaction_blocks < 0:
            raise ValueError("num_global_interaction_blocks must be non-negative")
        if global_num_time_units <= 0:
            raise ValueError("global_num_time_units must be positive")
        if interaction_fusion_init < 0:
            raise ValueError("interaction_fusion_init must be non-negative")

        self.dim = int(dim)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.n_fft = int(n_fft)
        self.num_layers = int(num_layers)
        self.num_sources = int(num_sources)
        self.compression_factor = float(compression_factor)
        self.mask_scale = float(mask_scale)
        self.reference_conditioning_mode = str(reference_conditioning_mode)
        self.film_scale = float(film_scale)
        self.lookahead_frames = int(lookahead_frames)
        self.use_speaker_encoder = bool(use_speaker_encoder)
        self.speaker_encoder_source = str(speaker_encoder_source)
        self.speaker_embedding_dim = int(speaker_embedding_dim)

        self.stft_encoder = CausalSTFTEncoder(
            win_length,
            hop_length,
            n_fft,
        )
        self.stft_decoder = CausalISTFTDecoder(
            win_length,
            hop_length,
            n_fft,
        )

        # Shared full-resolution embedding for mixture and reference-similarity.
        # Each branch emits dim channels, so the concatenated feature has 4*dim.
        self.tf_embedding = nn.Sequential(
            CausalMultiRangeEmbedding(2, dim, bias=False),
            TimeIndependentGroupNorm(1, dim * 4, eps=eps),
            nn.PReLU(dim * 4),
        )

        interaction_channels = dim * 4
        num_heads = 4 if dim % 4 == 0 else 1
        self.reference_conditioner = CausalLocalGlobalReferenceConditioner(  ################################ 最核心的创新模块
            interaction_channels=interaction_channels,
            output_channels=dim,
            num_local_blocks=num_interaction_blocks,
            num_global_blocks=num_global_interaction_blocks,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            global_num_time_units=global_num_time_units,
            layer_scale=layer_scale,
            eps=eps,
            mode=reference_conditioning_mode,
            fusion_init_scale=interaction_fusion_init,
        )

        # Frozen speaker encoder -> FiLM. The final FiLM layer is zero-initialized
        # so speaker modulation starts as an identity operation.
        self.speaker_film = nn.Sequential(
            nn.Linear(self.speaker_embedding_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )
        nn.init.zeros_(self.speaker_film[-1].weight)
        nn.init.zeros_(self.speaker_film[-1].bias)

        self.blocks = nn.ModuleList(
            [
                OnlineFreqTimeBlock(
                    channels=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    num_time_units=num_time_units,
                    layer_scale=layer_scale,
                )
                for _ in range(num_layers)
            ]
        )

        # Directly regress the compressed complex spectrum. Default random init,
        # matching the legacy online model's deconv output stage.
        self.output_head = CausalConv2dTF(
            dim,
            2 * num_sources,
            kernel_size=(3, 3),
            bias=True,
        )

        self.speaker_encoder = None
        if self.use_speaker_encoder:
            self._load_speaker_encoder(speaker_encoder_device)

    def _load_speaker_encoder(
        self,
        speaker_encoder_device: Optional[str],
    ) -> None:
        if EncoderClassifier is None:
            raise ImportError(
                "SpeechBrain is required when use_speaker_encoder=True. "
                "Install speechbrain or construct the model with "
                "use_speaker_encoder=False."
            )

        if speaker_encoder_device is None:
            if torch.cuda.is_available():
                speaker_encoder_device = f"cuda:{torch.cuda.current_device()}"
            else:
                speaker_encoder_device = "cpu"

        self.speaker_encoder = EncoderClassifier.from_hparams(
            source=self.speaker_encoder_source,
            savedir=None,
            run_opts={"device": str(speaker_encoder_device)},
            hparams_file="hyperparams.yaml",
        )
        self.speaker_encoder.to(torch.device(speaker_encoder_device))
        self.speaker_encoder.eval()

        for parameter in self.speaker_encoder.parameters():
            parameter.requires_grad = False

    @property
    def algorithmic_latency_samples(self) -> int:
        # Conservative frame-level latency used by the existing streaming wrapper.
        return (
            self.win_length - 1
            + self.lookahead_frames * self.hop_length
        )

    def algorithmic_latency_ms(self, sample_rate: int = 16000) -> float:
        return 1000.0 * self.algorithmic_latency_samples / sample_rate

    @staticmethod
    def _waveform_to_2d(wav: torch.Tensor) -> torch.Tensor:
        return CausalSTFTEncoder._to_2d(wav)

    def _reference_lengths_to_samples(
        self,
        lengths: Optional[torch.Tensor],
        batch_size: int,
        max_samples: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        支持两种输入：
        1. absolute sample length，例如 52341
        2. relative length，例如 0.73

        返回真实 sample 数：[B]
        """
        if lengths is None:
            return None

        lengths = torch.as_tensor(
            lengths,
            device=device,
        ).reshape(-1)

        if lengths.numel() != batch_size:
            raise ValueError(
                f"reference length batch mismatch: "
                f"{lengths.numel()} vs {batch_size}"
            )

        lengths_float = lengths.float()

        # 兼容旧 DataLoader 的 relative length
        if float(lengths_float.max().item()) <= 1.0001:
            sample_lengths = torch.round(
                lengths_float * max_samples
            ).long()
        else:
            sample_lengths = torch.round(
                lengths_float
            ).long()

        return sample_lengths.clamp(
            min=1,
            max=max_samples,
        )


    def get_speaker_embedding(
        self,
        aux: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.use_speaker_encoder:
            return None
        if self.speaker_encoder is None:
            raise RuntimeError("Speaker encoder has not been initialized")

        wav = self._waveform_to_2d(aux).float()
        device = wav.device

        # Keep SpeechBrain internal device bookkeeping synchronized with DDP rank.
        self.speaker_encoder.to(device)
        if hasattr(self.speaker_encoder, "device"):
            self.speaker_encoder.device = device
        if hasattr(self.speaker_encoder, "mods"):
            self.speaker_encoder.mods.to(device)

        self.speaker_encoder.eval()
        for parameter in self.speaker_encoder.parameters():
            parameter.requires_grad = False

        with torch.no_grad():
            embedding = self.speaker_encoder.encode_batch(
                wav.to(device).contiguous()
            )

        embedding = embedding.squeeze(1)
        embedding = F.normalize(embedding.float(), dim=-1)
        if embedding.shape[-1] != self.speaker_embedding_dim:
            raise RuntimeError(
                "Speaker embedding dimension mismatch: "
                f"expected {self.speaker_embedding_dim}, "
                f"got {embedding.shape[-1]}"
            )
        return embedding

    def compress_spectrum(self, spectrum_ri: torch.Tensor) -> torch.Tensor:
        spectrum_ri = spectrum_ri.float()
        spectrum = torch.complex(spectrum_ri[:, 0], spectrum_ri[:, 1])
        magnitude = spectrum.abs().clamp_min(1.0e-12).pow(
            self.compression_factor
        )
        phase = torch.angle(spectrum)
        output = torch.stack(
            [magnitude * phase.cos(), magnitude * phase.sin()],
            dim=1,
        )
        return torch.nan_to_num(output)

    def decompress_spectrum(self, spectrum_ri: torch.Tensor) -> torch.Tensor:
        spectrum_ri = spectrum_ri.float()
        spectrum = torch.complex(spectrum_ri[:, 0], spectrum_ri[:, 1])
        # clamp 上限与旧模型 FeaDecompression 一致（压缩域 1e4）
        magnitude = spectrum.abs().clamp(min=1.0e-12, max=1.0e4)
        magnitude = magnitude.pow(1.0 / self.compression_factor)
        phase = torch.angle(spectrum)
        output = torch.stack(
            [magnitude * phase.cos(), magnitude * phase.sin()],
            dim=1,
        )
        return torch.nan_to_num(
            output,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )

    @staticmethod
    def compute_similarity_batch(
        mixture: torch.Tensor,
        enrollment: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-attend each mixture frame to a fixed reference memory.

        Args:
            mixture:    [B,2,F,T_mix]
            enrollment: [B,2,F,T_ref]
        Returns:
            similarity: [B,2,F,T_mix]

        This is causal with respect to mixture time because no mixture future
        frame participates in another mixture frame's output.
        """
        mixture = mixture.float()
        enrollment = enrollment.float()

        if mixture.shape[0] != enrollment.shape[0]:
            raise ValueError(
                "Mixture/reference batch sizes must match: "
                f"{mixture.shape[0]} vs {enrollment.shape[0]}"
            )
        if mixture.shape[1] != 2 or enrollment.shape[1] != 2:
            raise ValueError("Expected real/imag channel dimension equal to 2")
        if mixture.shape[2] != enrollment.shape[2]:
            raise ValueError(
                "Mixture/reference frequency sizes must match: "
                f"{mixture.shape[2]} vs {enrollment.shape[2]}"
            )

        _, channels, frequency_bins, _ = mixture.shape
        scale = math.sqrt(float(frequency_bins))
        outputs = []

        for channel in range(channels):
            mix_channel = mixture[:, channel]       # [B,F,T_mix]
            ref_channel = enrollment[:, channel]    # [B,F,T_ref]

            scores = torch.bmm(
                ref_channel.transpose(1, 2),
                mix_channel,
            ) / scale                               # [B,T_ref,T_mix]
            scores = scores - scores.amax(dim=1, keepdim=True)
            attention = torch.softmax(scores, dim=1)
            attention = torch.nan_to_num(
                attention,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            attended = torch.bmm(
                ref_channel,
                attention,
            )                                       # [B,F,T_mix]
            outputs.append(attended.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    def encode_reference(
        self,
        aux: torch.Tensor,
        reference_lengths: Optional[torch.Tensor] = None,
    ) -> ReferenceConditioning:
        """Encode enrollment audio once before separation.

        If ``reference_lengths`` is omitted, the whole ``aux`` tensor is treated
        as valid audio. This preserves the original fixed-length inference and
        streaming behavior.

        If ``reference_lengths`` is provided, the batch may contain different
        enrollment durations. Each item is cropped to its true waveform length
        and encoded independently. No zero padding is therefore visible to the
        spectral reference memory or to ECAPA.
        """
        aux_2d = self._waveform_to_2d(aux)
        batch_size, padded_samples = aux_2d.shape

        sample_lengths = self._reference_lengths_to_samples(
            reference_lengths,
            batch_size=batch_size,
            max_samples=padded_samples,
            device=aux_2d.device,
        )

        # Legacy/fixed-length path: keep the original fully batched behavior.
        if sample_lengths is None:
            aux_spectrum, _ = self.stft_encoder(aux_2d)
            aux_compressed = self.compress_spectrum(aux_spectrum)
            speaker_embedding = self.get_speaker_embedding(aux_2d)
            return ReferenceConditioning(
                compressed_spectrum=aux_compressed,
                speaker_embedding=speaker_embedding,
                compressed_spectrum_list=None,
            )

        # Variable-length path: crop every enrollment before any reference
        # processing. This is intentionally simple and correctness-first.
        lengths_list = sample_lengths.detach().cpu().tolist()
        compressed_list: List[torch.Tensor] = []
        embedding_list: List[torch.Tensor] = []

        for batch_index, true_length in enumerate(lengths_list):
            true_length = int(true_length)
            aux_i = aux_2d[batch_index : batch_index + 1, :true_length]

            spectrum_i, _ = self.stft_encoder(aux_i)
            compressed_i = self.compress_spectrum(spectrum_i)
            compressed_list.append(compressed_i)

            embedding_i = self.get_speaker_embedding(aux_i)
            if embedding_i is not None:
                embedding_list.append(embedding_i)

        if self.use_speaker_encoder:
            if len(embedding_list) != batch_size:
                raise RuntimeError(
                    "Failed to compute one speaker embedding per reference: "
                    f"{len(embedding_list)} vs {batch_size}"
                )
            speaker_embedding: Optional[torch.Tensor] = torch.cat(
                embedding_list,
                dim=0,
            )
        else:
            speaker_embedding = None

        return ReferenceConditioning(
            compressed_spectrum=None,
            speaker_embedding=speaker_embedding,   #  TF-level info 
            compressed_spectrum_list=compressed_list,  # speaker-level info
        )

    def _speaker_film_parameters(
        self,
        conditioning: ReferenceConditioning,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conditioning.speaker_embedding is None:
            zeros = torch.zeros(
                batch_size,
                self.dim,
                1,
                1,
                device=device,
                dtype=dtype,
            )
            return zeros, zeros

        embedding = conditioning.speaker_embedding.to(
            device=device,
            dtype=self.speaker_film[0].weight.dtype,
        )
        film = self.speaker_film(embedding)
        gamma, beta = film.chunk(2, dim=-1)
        gamma = torch.tanh(gamma).view(batch_size, self.dim, 1, 1)
        beta = beta.view(batch_size, self.dim, 1, 1)
        return gamma.to(dtype=dtype), beta.to(dtype=dtype)

    def extract_conditioned_features(     ################################核心模块
        self,
        mixture_spectrum_ri: torch.Tensor,
        conditioning: ReferenceConditioning,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return separator features and compressed mixture spectrum.

        This helper is useful for causality tests because it observes the actual
        conditioned representation before the zero-initialized output head.
        """
        batch_size, _, _, _ = mixture_spectrum_ri.shape
        mixture_compressed = self.compress_spectrum(mixture_spectrum_ri)

        # Variable-length reference batches are represented as a list of
        # unpadded spectral memories. Cross-attention is computed per item, then
        # the resulting mixture-aligned features are concatenated back into a
        # normal batch. The expensive separator backbone below remains batched.
        if conditioning.compressed_spectrum_list is not None:
            reference_list = conditioning.compressed_spectrum_list
            if len(reference_list) != batch_size:
                raise ValueError(
                    "Mixture/reference batch size mismatch: "
                    f"{batch_size} vs {len(reference_list)}"
                )

            similarity_list: List[torch.Tensor] = []
            for batch_index, reference_i in enumerate(reference_list):
                reference_i = reference_i.to(
                    device=mixture_compressed.device,
                    dtype=mixture_compressed.dtype,
                )
                mixture_i = mixture_compressed[
                    batch_index : batch_index + 1
                ]

                similarity_i = self.compute_similarity_batch(
                    mixture_i.transpose(-2, -1).contiguous(),
                    reference_i.transpose(-2, -1).contiguous(),
                ).transpose(-2, -1).contiguous()
                similarity_list.append(similarity_i)

            similarity = torch.cat(similarity_list, dim=0)

        else:
            if conditioning.compressed_spectrum is None:
                raise RuntimeError(
                    "ReferenceConditioning contains neither a batched reference "
                    "spectrum nor a variable-length reference list"
                )

            reference_compressed = conditioning.compressed_spectrum.to(
                device=mixture_compressed.device,
                dtype=mixture_compressed.dtype,
            )
            if reference_compressed.shape[0] != batch_size:
                raise ValueError(
                    "Mixture/reference batch size mismatch: "
                    f"{batch_size} vs {reference_compressed.shape[0]}"
                )

            similarity = self.compute_similarity_batch(
                mixture_compressed.transpose(-2, -1).contiguous(),
                reference_compressed.transpose(-2, -1).contiguous(),
            ).transpose(-2, -1).contiguous()

        mixture_features = self.tf_embedding(mixture_compressed)
        reference_features = self.tf_embedding(similarity)

        features = self.reference_conditioner(
            mixture_features,
            reference_features,
        )

        gamma, beta = self._speaker_film_parameters(
            conditioning,
            batch_size=batch_size,
            device=features.device,
            dtype=features.dtype,
        )

        # Speaker conditioning only once
        features = (
            features * (1.0 + self.film_scale * gamma)
            + self.film_scale * beta
        )

        for block in self.blocks:
            features = block(features)

        return features, mixture_compressed

    def forward_spectrum_with_reference(
        self,
        mixture_spectrum_ri: torch.Tensor,
        conditioning: ReferenceConditioning,
    ) -> torch.Tensor:
        batch_size, _, frames, frequencies = mixture_spectrum_ri.shape
        features, _ = self.extract_conditioned_features(  # 先做compression，也就是保证mixture和reference都是压缩的，以及计算相似度
            mixture_spectrum_ri,
            conditioning,
        )
        # 直接回归压缩复频谱，与旧模型 mugi_block_plus_v2_online 的输出通路保持一致，
        # 保证消融实验中两个模型只差 conditioning 模块
        target_compressed = self.output_head(features)
        target_spectrum = self.decompress_spectrum(target_compressed)

        return target_spectrum.view(
            batch_size,
            self.num_sources,
            2,
            frames,
            frequencies,
        )

    def forward_with_reference_cache(
        self,
        x: torch.Tensor,
        conditioning: ReferenceConditioning,
    ) -> torch.Tensor:
        mixture_spectrum, meta = self.stft_encoder(x)  # [B, 2, Tmix, F]
        target_spectrum = self.forward_spectrum_with_reference(
            mixture_spectrum,
            conditioning,
        )
        return self.stft_decoder(target_spectrum, meta)

    def forward(
        self,
        x: torch.Tensor,
        aux: torch.Tensor,
        reference_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract the target speaker.

        Args:
            x: Mixture waveform [B,T] or [B,1,T].
            aux: Padded enrollment waveform [B,R] or [B,1,R].
            reference_lengths: Optional true enrollment lengths [B]. Pass 每个reference的真实长度
                absolute sample counts for the online DataLoader. Relative
                lengths in (0,1] are also accepted for legacy loaders.
        """
        conditioning = self.encode_reference(  # 先把reference编码好，然后再处理mixture，作为缓存，实时计算时，不需要每一个mixture chunk都重新跑一次ECAPA以及stft
            aux,
            reference_lengths=reference_lengths,
        )
        return self.forward_with_reference_cache(x, conditioning)


# Backward-compatible alias for training configurations that still import the
# previous class name. Public forward also accepts optional reference_lengths.
OnlineSpeechEnhancementModel = OnlineTargetSpeakerExtractionModel
mugi_block_plus_v2_online = OnlineTargetSpeakerExtractionModel

# -----------------------------------------------------------------------------
# Stateful fixed-context online wrapper.
# It never uses samples after the current chunk. Samples are released only after
# the model's STFT lookahead has elapsed.
# -----------------------------------------------------------------------------
class StreamingTargetSpeakerExtractor:
    """
    Fixed-context streaming wrapper.

    The reference is encoded once with set_reference()/constructor and reused.
    Mixture history is still recomputed inside a fixed context window; this is
    online and causal, but not yet per-layer state-cached incremental Mamba.
    """

    def __init__(
        self,
        model: OnlineTargetSpeakerExtractionModel,
        aux: torch.Tensor,
        sample_rate: int = 16000,
        context_seconds: float = 2.0,
    ):
        if context_seconds <= 0:
            raise ValueError("context_seconds must be positive")

        self.model = model
        self.sample_rate = int(sample_rate)
        self.context_samples = max(
            model.win_length,
            int(round(context_seconds * sample_rate)),
        )
        self.reference: Optional[ReferenceConditioning] = None
        self.set_reference(aux)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.inference_mode()
    def set_reference(self, aux: torch.Tensor) -> None:
        aux = CausalSTFTEncoder._to_2d(aux).to(self.device)
        self.reference = self.model.encode_reference(aux)
        self.reset_stream()

    def reset_stream(self) -> None:
        self.buffer: Optional[torch.Tensor] = None
        self.buffer_start = 0
        self.total_received = 0
        self.total_emitted = 0

    @torch.inference_mode()
    def process_chunk(
        self,
        chunk: torch.Tensor,
        final: bool = False,
    ) -> torch.Tensor:
        if self.reference is None:
            raise RuntimeError("Reference has not been set")

        chunk_2d = CausalSTFTEncoder._to_2d(chunk).to(self.device)
        if self.buffer is None:
            self.buffer = chunk_2d
        else:
            if chunk_2d.shape[0] != self.buffer.shape[0]:
                raise ValueError(
                    "Batch size must remain fixed within one stream"
                )
            self.buffer = torch.cat([self.buffer, chunk_2d], dim=-1)

        self.total_received += chunk_2d.shape[-1]
        latency = self.model.algorithmic_latency_samples
        available_until = (
            self.total_received
            if final
            else max(0, self.total_received - latency)
        )

        if available_until <= self.total_emitted:
            return chunk_2d.new_zeros(chunk_2d.shape[0], 1, 0)

        extracted_buffer = self.model.forward_with_reference_cache(
            self.buffer,
            self.reference,
        )

        local_start = self.total_emitted - self.buffer_start
        local_end = available_until - self.buffer_start
        if local_start < 0 or local_end > extracted_buffer.shape[-1]:
            raise RuntimeError(
                "Streaming buffer bookkeeping failed. Increase "
                "context_seconds or reset the stream."
            )

        output = extracted_buffer[..., local_start:local_end]
        self.total_emitted = available_until

        keep_from_global = max(
            0,
            self.total_emitted - self.context_samples,
        )
        drop = keep_from_global - self.buffer_start
        if drop > 0:
            self.buffer = self.buffer[..., drop:].contiguous()
            self.buffer_start = keep_from_global

        if final:
            self.reset_stream()
        return output

    @torch.inference_mode()
    def flush(self) -> torch.Tensor:
        if self.buffer is None:
            return torch.empty(1, 1, 0, device=self.device)
        batch_size = self.buffer.shape[0]
        empty = self.buffer.new_zeros(batch_size, 0)
        return self.process_chunk(empty, final=True)


# Backward-compatible alias.
StreamingSpeechEnhancer = StreamingTargetSpeakerExtractor

# -----------------------------------------------------------------------------
# Complexity and latency utilities.
# -----------------------------------------------------------------------------


@torch.inference_mode()
def verify_spectral_prefix_causality(
    model: OnlineTargetSpeakerExtractionModel,
    frames: int = 8,
    reference_frames: int = 6,
    frequencies: int = 17,
    cut_frame: int = 4,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> Dict[str, float]:
    """Numerically verify that future mixture frames cannot change past features.

    The test creates two mixture spectra with an identical prefix and completely
    different suffixes. It then compares the conditioned separator features
    before ``cut_frame``. The fixed reference memory is identical in both runs.

    This checks the real spectral backbone rather than the waveform output,
    because the output head is intentionally initialized to zero and could hide
    an internal future-information leak at initialization.
    """
    if not 1 <= cut_frame < frames:
        raise ValueError("cut_frame must satisfy 1 <= cut_frame < frames")

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(12345)

    prefix = torch.randn(
        1,
        2,
        cut_frame,
        frequencies,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    future_a = torch.randn(
        1,
        2,
        frames - cut_frame,
        frequencies,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    future_b = torch.randn(
        1,
        2,
        frames - cut_frame,
        frequencies,
        device=device,
        dtype=dtype,
        generator=generator,
    ) * 3.0

    mixture_a = torch.cat([prefix, future_a], dim=2)
    mixture_b = torch.cat([prefix, future_b], dim=2)
    reference = torch.randn(
        1,
        2,
        reference_frames,
        frequencies,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    conditioning = ReferenceConditioning(
        compressed_spectrum=reference,
        speaker_embedding=None,
    )

    features_a, _ = model.extract_conditioned_features(
        mixture_a,
        conditioning,
    )
    features_b, _ = model.extract_conditioned_features(
        mixture_b,
        conditioning,
    )

    prefix_a = features_a[:, :, :cut_frame]
    prefix_b = features_b[:, :, :cut_frame]
    absolute_error = (prefix_a - prefix_b).abs()
    max_abs_error = float(absolute_error.max().item())
    mean_abs_error = float(absolute_error.mean().item())
    passed = torch.allclose(prefix_a, prefix_b, atol=atol, rtol=rtol)

    if not passed:
        raise AssertionError(
            "Causality test failed: changing future mixture frames altered "
            f"past conditioned features; max_abs_error={max_abs_error:.3e}"
        )

    return {
        "passed": 1.0,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cut_frame": float(cut_frame),
    }


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def estimate_macs(
    model: nn.Module,
    mixture: torch.Tensor,
    aux: torch.Tensor,
) -> Dict[str, float]:
    counts = {
        "conv": 0.0,
        "linear": 0.0,
        "attention": 0.0,
        "scan": 0.0,
    }
    hooks = []

    def conv1d_hook(module: nn.Conv1d, inputs, output):
        out = output if isinstance(output, torch.Tensor) else output[0]
        kernel = module.kernel_size[0]
        counts["conv"] += float(
            out.numel()
            * (module.in_channels // module.groups)
            * kernel
        )

    def conv2d_hook(module: nn.Conv2d, inputs, output):
        out = output if isinstance(output, torch.Tensor) else output[0]
        kernel_h, kernel_w = module.kernel_size
        counts["conv"] += float(
            out.numel()
            * (module.in_channels // module.groups)
            * kernel_h
            * kernel_w
        )

    def linear_hook(module: nn.Linear, inputs, output):
        out = output if isinstance(output, torch.Tensor) else output[0]
        counts["linear"] += float(out.numel() * module.in_features)

    def attention_hook(module: Attention, inputs, output):
        x = inputs[0]
        batch, length, channels = x.shape
        counts["attention"] += float(
            2 * batch * length * length * channels
        )

    def scan_hook(module: UniMambaTemporalMixer, inputs, output):
        x = inputs[0]
        batch, length, _ = x.shape
        counts["scan"] += float(
            4
            * batch
            * length
            * module.d_inner
            * module.d_state
        )

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv2d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, Attention):
            hooks.append(module.register_forward_hook(attention_hook))
        elif isinstance(module, UniMambaTemporalMixer):
            hooks.append(module.register_forward_hook(scan_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            _ = model(mixture, aux)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    counts["total_macs"] = sum(counts.values())
    counts["estimated_flops"] = 2.0 * counts["total_macs"]
    return counts


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_reference_encoding(
    model: OnlineTargetSpeakerExtractionModel,
    aux: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()

    for _ in range(warmup):
        _ = model.encode_reference(aux)
    _synchronize(device)

    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.encode_reference(aux)
        _synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    return {
        "mean_ms": statistics.mean(times_ms),
        "p50_ms": statistics.median(times_ms),
        "p95_ms": sorted(times_ms)[
            max(0, math.ceil(0.95 * len(times_ms)) - 1)
        ],
    }


@torch.inference_mode()
def benchmark_full_utterance(
    model: OnlineTargetSpeakerExtractionModel,
    sample_rate: int,
    seconds: float,
    aux: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    samples = max(
        model.win_length,
        int(round(sample_rate * seconds)),
    )
    mixture = torch.randn(1, 1, samples, device=device)
    reference = model.encode_reference(aux)
    model.eval()

    for _ in range(warmup):
        _ = model.forward_with_reference_cache(mixture, reference)
    _synchronize(device)

    times_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        _ = model.forward_with_reference_cache(mixture, reference)
        _synchronize(device)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(times_ms)
    return {
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(times_ms),
        "p95_ms": sorted(times_ms)[
            max(0, math.ceil(0.95 * len(times_ms)) - 1)
        ],
        "rtf": mean_ms / (1000.0 * seconds),
    }


@torch.inference_mode()
def benchmark_streaming(
    model: OnlineTargetSpeakerExtractionModel,
    aux: torch.Tensor,
    sample_rate: int,
    chunk_ms: float,
    context_seconds: float,
    num_chunks: int,
    warmup_chunks: int,
) -> Dict[str, float]:
    device = next(model.parameters()).device
    chunk_samples = max(
        1,
        int(round(sample_rate * chunk_ms / 1000.0)),
    )
    streamer = StreamingTargetSpeakerExtractor(
        model,
        aux,
        sample_rate,
        context_seconds,
    )
    model.eval()

    times_ms: List[float] = []
    for index in range(warmup_chunks + num_chunks):
        chunk = torch.randn(1, chunk_samples, device=device)
        start = time.perf_counter()
        _ = streamer.process_chunk(chunk)
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if index >= warmup_chunks:
            times_ms.append(elapsed_ms)

    mean_ms = statistics.mean(times_ms)
    return {
        "chunk_samples": float(chunk_samples),
        "mean_ms": mean_ms,
        "p50_ms": statistics.median(times_ms),
        "p95_ms": sorted(times_ms)[
            max(0, math.ceil(0.95 * len(times_ms)) - 1)
        ],
        "chunk_rtf": mean_ms / chunk_ms,
    }


def human_number(value: float) -> str:
    for scale, suffix in (
        (1.0e12, "T"),
        (1.0e9, "G"),
        (1.0e6, "M"),
        (1.0e3, "K"),
    ):
        if abs(value) >= scale:
            return f"{value / scale:.3f}{suffix}"
    return f"{value:.3f}"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Online target-speaker-extraction benchmark"
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-time-units", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--d-state", type=int, default=16)
    parser.add_argument("--d-conv", type=int, default=3)
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--compression-factor", type=float, default=0.5)
    parser.add_argument("--mask-scale", type=float, default=1.0)
    parser.add_argument("--num-interaction-blocks", type=int, default=2)
    parser.add_argument("--num-global-interaction-blocks", type=int, default=1)
    parser.add_argument("--global-num-time-units", type=int, default=1)
    parser.add_argument(
        "--reference-conditioning-mode",
        type=str,
        choices=["local", "global", "local_global"],
        default="local_global",
    )
    parser.add_argument("--interaction-fusion-init", type=float, default=1.0e-3)
    parser.add_argument("--film-scale", type=float, default=0.2)
    parser.add_argument("--win-length", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--reference-seconds", type=float, default=1.0)
    parser.add_argument("--benchmark-seconds", type=float, default=1.0)
    parser.add_argument("--complexity-seconds", type=float, default=1.0)
    parser.add_argument("--chunk-ms", type=float, default=100.0)
    parser.add_argument("--context-seconds", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--stream-chunks", type=int, default=20)
    parser.add_argument(
        "--speaker-encoder-source",
        type=str,
        default=(
            "/home/xueke/real_tse_challenge/"
            "spkrec-ecapa-voxceleb"
        ),
    )
    parser.add_argument(
        "--disable-speaker-encoder",
        action="store_true",
        help=(
            "Disable frozen ECAPA/FiLM; spectral reference similarity "
            "conditioning remains enabled."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--skip-causality-test", action="store_true")
    parser.add_argument("--skip-complexity", action="store_true")
    parser.add_argument("--skip-runtime", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    # ============================================================
    # 1. Build model
    # ============================================================
    model = OnlineTargetSpeakerExtractionModel(
        dim=args.dim,
        mlp_ratio=args.mlp_ratio,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_fft=args.n_fft,
        num_layers=args.num_layers,
        d_state=args.d_state,
        d_conv=args.d_conv,
        expand=args.expand,
        num_time_units=args.num_time_units,
        num_sources=1,
        compression_factor=args.compression_factor,
        mask_scale=args.mask_scale,
        num_interaction_blocks=args.num_interaction_blocks,
        num_global_interaction_blocks=args.num_global_interaction_blocks,
        global_num_time_units=args.global_num_time_units,
        reference_conditioning_mode=args.reference_conditioning_mode,
        interaction_fusion_init=args.interaction_fusion_init,
        film_scale=args.film_scale,
        layer_scale=1.0e-5,
        eps=1.0e-5,
        lookahead_frames=0,
        use_speaker_encoder=not args.disable_speaker_encoder,
        speaker_encoder_source=args.speaker_encoder_source,
        speaker_embedding_dim=192,
        speaker_encoder_device=str(device),
    ).to(device)

    model.eval()

    total_params, trainable_params = count_parameters(model)

    print("\n")
    print("=" * 90)
    print("OnlineTargetSpeakerExtractionModel")
    print("Variable-Length Reference Batch Validation")
    print("=" * 90)

    print(f"Device                       : {device}")
    print(f"Fused selective scan         : {HAS_FUSED_SELECTIVE_SCAN}")
    print(f"Speaker encoder enabled      : {model.use_speaker_encoder}")
    print(
        f"Reference condition mode     : "
        f"{model.reference_conditioning_mode}"
    )

    print(
        f"Parameters                   : "
        f"{human_number(total_params)} "
        f"({total_params:,})"
    )

    print(
        f"Trainable parameters         : "
        f"{human_number(trainable_params)} "
        f"({trainable_params:,})"
    )

    print(
        f"Algorithmic latency          : "
        f"{model.algorithmic_latency_ms(args.sample_rate):.3f} ms "
        f"({model.algorithmic_latency_samples} samples)"
    )

    print("=" * 90)


    # ============================================================
    # Small SI-SDR helper for identity test
    # ============================================================
    def test_sisdr(
        reference: torch.Tensor,
        estimate: torch.Tensor,
        eps: float = 1.0e-8,
    ) -> torch.Tensor:

        reference = (
            reference
            - reference.mean(
                dim=-1,
                keepdim=True,
            )
        )

        estimate = (
            estimate
            - estimate.mean(
                dim=-1,
                keepdim=True,
            )
        )

        reference_energy = (
            reference.pow(2)
            .sum(
                dim=-1,
                keepdim=True,
            )
            + eps
        )

        scale = (
            (estimate * reference)
            .sum(
                dim=-1,
                keepdim=True,
            )
            / reference_energy
        )

        target = scale * reference
        noise = estimate - target

        ratio = (
            target.pow(2).sum(dim=-1)
            /
            (
                noise.pow(2).sum(dim=-1)
                + eps
            )
        )

        return 10.0 * torch.log10(
            ratio + eps
        )


    # ============================================================
    # 2. Variable-reference batch test
    #
    # Simulate exactly what the online DataLoader does:
    #
    # reference 0 = 3 s
    # reference 1 = 5 s
    # reference 2 = 8 s
    # reference 3 = 12 s
    #
    # All are padded to 12 s in one batch.
    # ============================================================

    batch_size = 4

    reference_seconds_list = [
        3.0,
        5.0,
        8.0,
        12.0,
    ]

    reference_lengths = torch.tensor(
        [
            int(
                round(
                    seconds
                    * args.sample_rate
                )
            )
            for seconds in reference_seconds_list
        ],
        dtype=torch.long,
        device=device,
    )

    max_reference_samples = int(
        reference_lengths.max().item()
    )

    # Use a 3-second training mixture,
    # matching your online training configuration.
    mixture_samples = int(
        round(
            3.0
            * args.sample_rate
        )
    )

    # ------------------------------------------------------------
    # Mixture batch
    # [B, 1, 48000]
    # ------------------------------------------------------------
    mixture = torch.randn(
        batch_size,
        1,
        mixture_samples,
        device=device,
    ) * 0.05

    # ------------------------------------------------------------
    # Padded reference batch
    # [B, 1, max_reference_samples]
    # ------------------------------------------------------------
    aux = torch.zeros(
        batch_size,
        1,
        max_reference_samples,
        device=device,
    )

    # Give every reference independent random speech-like input.
    # Important: only write into its real region.
    for index, true_length in enumerate(
        reference_lengths.tolist()
    ):

        aux[
            index,
            0,
            :true_length,
        ] = (
            torch.randn(
                true_length,
                device=device,
            )
            * 0.05
        )

    print("\n")
    print("=" * 90)
    print("[TEST 1] Variable-length reference batch")
    print("=" * 90)

    print(
        "Mixture shape                :",
        tuple(mixture.shape),
    )

    print(
        "Padded reference shape       :",
        tuple(aux.shape),
    )

    print(
        "Reference lengths (samples)  :",
        reference_lengths.detach().cpu().tolist(),
    )

    print(
        "Reference lengths (seconds)  :",
        [
            round(
                length / args.sample_rate,
                3,
            )
            for length
            in reference_lengths.detach().cpu().tolist()
        ],
    )

    print(
        "Padded reference duration    :",
        f"{max_reference_samples / args.sample_rate:.3f} s",
    )


    # ============================================================
    # 3. Directly inspect variable reference encoding
    # ============================================================

    with torch.inference_mode():

        conditioning = model.encode_reference(
            aux,
            reference_lengths=reference_lengths,
        )

    print("\n")
    print("-" * 90)
    print("[TEST 2] Internal reference encoding")
    print("-" * 90)

    if (
        conditioning.compressed_spectrum_list
        is None
    ):
        raise RuntimeError(
            "Expected variable-length "
            "compressed_spectrum_list, "
            "but got None."
        )

    if (
        len(
            conditioning.compressed_spectrum_list
        )
        != batch_size
    ):
        raise RuntimeError(
            "Reference list batch size mismatch."
        )

    for index, reference_memory in enumerate(
        conditioning.compressed_spectrum_list
    ):

        expected_frames = (
            model.stft_encoder.num_frames(
                int(
                    reference_lengths[index].item()
                )
            )
        )

        actual_frames = (
            reference_memory.shape[2]
        )

        print(
            f"Reference {index}: "
            f"{reference_seconds_list[index]:5.1f} s | "
            f"samples={reference_lengths[index].item():6d} | "
            f"memory={tuple(reference_memory.shape)} | "
            f"frames={actual_frames:4d} | "
            f"expected={expected_frames:4d}"
        )

        if actual_frames != expected_frames:
            raise RuntimeError(
                f"Reference {index} frame mismatch: "
                f"{actual_frames} != {expected_frames}"
            )

    print(
        "\nVariable reference STFT frame test: PASSED"
    )


    # ============================================================
    # 4. Check ECAPA speaker embedding batch
    # ============================================================

    if model.use_speaker_encoder:

        if conditioning.speaker_embedding is None:
            raise RuntimeError(
                "Speaker encoder enabled but "
                "speaker_embedding is None."
            )

        print(
            "Speaker embedding shape      :",
            tuple(
                conditioning
                .speaker_embedding
                .shape
            ),
        )

        if (
            conditioning
            .speaker_embedding
            .shape[0]
            != batch_size
        ):
            raise RuntimeError(
                "Speaker embedding batch "
                "size mismatch."
            )

        print(
            "ECAPA variable-length test    : PASSED"
        )

    else:

        print(
            "ECAPA test                    : SKIPPED "
            "(speaker encoder disabled)"
        )


    # ============================================================
    # 5. Actual forward test
    # ============================================================

    print("\n")
    print("-" * 90)
    print("[TEST 3] Full variable-length batch forward")
    print("-" * 90)

    with torch.inference_mode():

        output = model(
            mixture,
            aux,
            reference_lengths,
        )

    print(
        "Input mixture shape           :",
        tuple(mixture.shape),
    )

    print(
        "Output shape                  :",
        tuple(output.shape),
    )

    expected_output_shape = (
        batch_size,
        1,
        mixture_samples,
    )

    if tuple(output.shape) != expected_output_shape:

        raise RuntimeError(
            f"Output shape mismatch: "
            f"{tuple(output.shape)} "
            f"!= {expected_output_shape}"
        )

    print(
        "Variable batch forward        : PASSED"
    )


    # ============================================================
    # 6. Identity initialization check
    #
    # output_head is zero initialized,
    # so initial output should be approximately mixture.
    # ============================================================

    print("\n")
    print("-" * 90)
    print("[TEST 4] Initial identity-mask check")
    print("-" * 90)

    mixture_2d = mixture[:, 0]
    output_2d = output[:, 0]

    identity_error = (
        output_2d
        - mixture_2d
    )

    identity_mae = (
        identity_error
        .abs()
        .mean()
        .item()
    )

    identity_max_error = (
        identity_error
        .abs()
        .max()
        .item()
    )

    identity_sisdr = test_sisdr(
        mixture_2d,
        output_2d,
    )

    output_head_weight_max = (
        model.output_head
        .conv.weight
        .detach()
        .abs()
        .max()
        .item()
    )

    if (
        model.output_head.conv.bias
        is not None
    ):
        output_head_bias_max = (
            model.output_head
            .conv.bias
            .detach()
            .abs()
            .max()
            .item()
        )
    else:
        output_head_bias_max = 0.0

    print(
        f"Output-mixture MAE           : "
        f"{identity_mae:.8e}"
    )

    print(
        f"Output-mixture max error     : "
        f"{identity_max_error:.8e}"
    )

    print(
        "Identity SI-SDR (dB)         :",
        identity_sisdr.detach().cpu(),
    )

    print(
        f"Output head weight abs max   : "
        f"{output_head_weight_max:.8e}"
    )

    print(
        f"Output head bias abs max     : "
        f"{output_head_bias_max:.8e}"
    )

    if identity_mae > 1.0e-5:

        print(
            "WARNING: initial output is not "
            "very close to mixture."
        )

    else:

        print(
            "Initial identity test        : PASSED"
        )


    # ============================================================
    # 7. Padding invariance test
    #
    # Same real references,
    # but artificially pad another 3 seconds.
    #
    # Because reference_lengths is supplied,
    # encoded references MUST remain unchanged.
    # ============================================================

    print("\n")
    print("-" * 90)
    print("[TEST 5] Reference padding invariance")
    print("-" * 90)

    extra_padding_samples = int(
        round(
            3.0
            * args.sample_rate
        )
    )

    aux_more_padding = F.pad(
        aux,
        (
            0,
            extra_padding_samples,
        ),
        value=0.0,
    )

    with torch.inference_mode():

        conditioning_more_padding = (
            model.encode_reference(
                aux_more_padding,
                reference_lengths=reference_lengths,
            )
        )

    if (
        conditioning_more_padding
        .compressed_spectrum_list
        is None
    ):
        raise RuntimeError(
            "Expected variable reference list."
        )

    reference_memory_max_diff = 0.0

    for index in range(batch_size):

        memory_a = (
            conditioning
            .compressed_spectrum_list[index]
        )

        memory_b = (
            conditioning_more_padding
            .compressed_spectrum_list[index]
        )

        diff = (
            memory_a
            - memory_b
        ).abs().max().item()

        reference_memory_max_diff = max(
            reference_memory_max_diff,
            diff,
        )

        print(
            f"Reference {index} spectral "
            f"memory max diff: {diff:.8e}"
        )

    print(
        f"Overall spectral max diff    : "
        f"{reference_memory_max_diff:.8e}"
    )

    if model.use_speaker_encoder:

        embedding_diff = (
            conditioning
            .speaker_embedding
            - conditioning_more_padding
            .speaker_embedding
        ).abs().max().item()

        print(
            f"ECAPA embedding max diff     : "
            f"{embedding_diff:.8e}"
        )

    else:

        embedding_diff = 0.0

    # Because we crop BEFORE encoding,
    # this should normally be exactly zero.
    if (
        reference_memory_max_diff
        > 1.0e-6
        or embedding_diff > 1.0e-6
    ):

        raise RuntimeError(
            "Padding invariance failed. "
            "Extra zero padding changed "
            "reference encoding."
        )

    print(
        "Reference padding invariance  : PASSED"
    )


    # ============================================================
    # 8. Absolute-length vs relative-length test
    # ============================================================

    print("\n")
    print("-" * 90)
    print("[TEST 6] Absolute vs relative reference lengths")
    print("-" * 90)

    relative_lengths = (
        reference_lengths.float()
        / float(max_reference_samples)
    )

    with torch.inference_mode():

        conditioning_relative = (
            model.encode_reference(
                aux,
                reference_lengths=relative_lengths,
            )
        )

    relative_max_diff = 0.0

    for index in range(batch_size):

        memory_absolute = (
            conditioning
            .compressed_spectrum_list[index]
        )

        memory_relative = (
            conditioning_relative
            .compressed_spectrum_list[index]
        )

        diff = (
            memory_absolute
            - memory_relative
        ).abs().max().item()

        relative_max_diff = max(
            relative_max_diff,
            diff,
        )

    print(
        "Relative lengths              :",
        relative_lengths.detach().cpu(),
    )

    print(
        f"Absolute-relative max diff    : "
        f"{relative_max_diff:.8e}"
    )

    if relative_max_diff > 1.0e-6:
        raise RuntimeError(
            "Absolute/relative length "
            "conversion mismatch."
        )

    print(
        "Absolute/relative test        : PASSED"
    )


    # ============================================================
    # 9. Causality test
    # ============================================================

    if not args.skip_causality_test:

        print("\n")
        print("-" * 90)
        print("[TEST 7] Spectral causality")
        print("-" * 90)

        causality = (
            verify_spectral_prefix_causality(
                model
            )
        )

        print(
            "Spectral causality test      : "
            f"passed=True, "
            f"max_abs_error="
            f"{causality['max_abs_error']:.3e}"
        )


    # ============================================================
    # 10. Original complexity benchmark
    #
    # Keep this batch=1 because it is a per-stream benchmark.
    # ============================================================

    reference_samples = max(
        args.win_length,
        int(
            round(
                args.reference_seconds
                * args.sample_rate
            )
        ),
    )

    aux_single = torch.randn(
        1,
        1,
        reference_samples,
        device=device,
    )

    if not args.skip_complexity:

        print("\n")
        print("-" * 90)
        print("[BENCHMARK] Complexity")
        print("-" * 90)

        complexity_samples = max(
            args.win_length,
            int(
                round(
                    args.complexity_seconds
                    * args.sample_rate
                )
            ),
        )

        complexity_mixture = torch.randn(
            1,
            1,
            complexity_samples,
            device=device,
        )

        counts = estimate_macs(
            model,
            complexity_mixture,
            aux_single,
        )

        example_frames = (
            model.stft_encoder.num_frames(
                complexity_samples
            )
        )

        one_second_frames = (
            model.stft_encoder.num_frames(
                args.sample_rate
            )
        )

        scale_to_one_second = (
            one_second_frames
            / example_frames
        )

        print(
            f"Complexity mixture           : "
            f"{complexity_samples} samples"
        )

        print(
            f"Reference length             : "
            f"{reference_samples} samples"
        )

        print(
            f"Estimated MACs/input         : "
            f"{human_number(counts['total_macs'])}"
        )

        print(
            f"Estimated FLOPs/input        : "
            f"{human_number(counts['estimated_flops'])}"
        )

        print(
            "Approx. mixture MACs/s      : "
            f"{human_number(counts['total_macs'] * scale_to_one_second)}"
        )

        print(
            "MAC breakdown               : "
            f"conv={human_number(counts['conv'])}, "
            f"linear={human_number(counts['linear'])}, "
            f"attention={human_number(counts['attention'])}, "
            f"scan={human_number(counts['scan'])}"
        )


    # ============================================================
    # 11. Original runtime benchmark
    # ============================================================

    if not args.skip_runtime:

        print("\n")
        print("-" * 90)
        print("[BENCHMARK] Runtime")
        print("-" * 90)

        if (
            device.type == "cpu"
            and not HAS_FUSED_SELECTIVE_SCAN
        ):

            warnings.warn(
                "Runtime benchmark uses the slow "
                "selective-scan fallback on CPU.",
                RuntimeWarning,
            )

        reference_stats = (
            benchmark_reference_encoding(
                model,
                aux_single,
                args.warmup,
                args.repeats,
            )
        )

        full_stats = (
            benchmark_full_utterance(
                model,
                args.sample_rate,
                args.benchmark_seconds,
                aux_single,
                args.warmup,
                args.repeats,
            )
        )

        stream_stats = benchmark_streaming(
            model,
            aux_single,
            args.sample_rate,
            args.chunk_ms,
            args.context_seconds,
            args.stream_chunks,
            args.warmup,
        )

        print(
            "Reference encoding           : "
            f"mean={reference_stats['mean_ms']:.3f} ms, "
            f"p50={reference_stats['p50_ms']:.3f} ms, "
            f"p95={reference_stats['p95_ms']:.3f} ms"
        )

        print(
            "Full-utterance latency       : "
            f"mean={full_stats['mean_ms']:.3f} ms, "
            f"p50={full_stats['p50_ms']:.3f} ms, "
            f"p95={full_stats['p95_ms']:.3f} ms"
        )

        print(
            f"Full-utterance RTF           : "
            f"{full_stats['rtf']:.4f}"
        )

        print(
            "Streaming chunk latency      : "
            f"mean={stream_stats['mean_ms']:.3f} ms, "
            f"p50={stream_stats['p50_ms']:.3f} ms, "
            f"p95={stream_stats['p95_ms']:.3f} ms"
        )

        print(
            f"Streaming chunk RTF          : "
            f"{stream_stats['chunk_rtf']:.4f}"
        )

        print(
            "Real-time capable            : "
            f"{stream_stats['chunk_rtf'] < 1.0}"
        )


    # ============================================================
    # Final summary
    # ============================================================

    print("\n")
    print("=" * 90)
    print("ALL MODEL CHECKS FINISHED")
    print("=" * 90)

    print(
        "Variable reference batch      : PASS"
    )

    print(
        "Reference real-length cropping: PASS"
    )

    print(
        "Padding invariance             : PASS"
    )

    print(
        "Absolute/relative length       : PASS"
    )

    print(
        "Output shape                   : PASS"
    )

    print(
        "Initial identity mapping       : "
        f"{'PASS' if identity_mae <= 1.0e-5 else 'WARNING'}"
    )

    print("=" * 90)
    print("\n")


if __name__ == "__main__":      
    main()  