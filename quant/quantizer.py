"""
quantizer.py
=================================================

Uniform symmetric quantization (the paper's "Direct" method), plus the
right-shift + shifting-error machinery that eDQA builds on.

Reproduction notes (读的时候特别注意):
  * scale 是 **逐通道 (per-channel)** 的 |max|，不是整层共用一个标量。这是跟
    论文作者直接确认过的（2026-07 邮件往来）："And yes, it's per channel
    quantization." Algorithm 1/2 写的 |max(A_layer)| 字面上容易读成整层共享，
    但实际实现是每个 channel 各自算自己的 max/scale。见 `compute_scale` 的
    `channel_dim` 参数（默认 1，即逐通道）；传 `channel_dim=None` 可以切回旧的
    整层字面实现，用于对比/消融。
  * 论文分母用 2^(N-1); 更常见的对称量化写法是 2^(N-1)-1。这里默认严格照论文
    (SCALE_DENOM_MINUS_ONE=False), 想切换成标准写法把它设 True。
  * shifting error 的语义 = "读取低 m 位再映射成小数", 与 Round(I/Δ_{N+m}) 右移
    m 位一致 (见 Eq.(3))。低 m 位整数 k -> 小数 k / 2^m, 落在 [0, 1)。
    m=3 时正好是 {0, 0.125, ..., 0.875}, 即论文 Figure 2 的横轴。
"""

from __future__ import annotations

import torch

# 论文严格写法用 2^(N-1); 设 True 切成标准对称量化 2^(N-1)-1
SCALE_DENOM_MINUS_ONE = False


def _denom(n_bits: int) -> float:
    return (2 ** (n_bits - 1) - 1) if SCALE_DENOM_MINUS_ONE else (2 ** (n_bits - 1))


def compute_scale(
    x: torch.Tensor, n_bits: int, channel_dim: int | None = 1, clip_percentile: float | None = None
):
    """Symmetric quantization scale Δ_N = |max| / 2^(N-1).

    Per-channel by default (channel_dim=1): each channel gets its own max_abs/
    scale, confirmed correct by the paper's authors. Pass channel_dim=None for
    the literal Algorithm-1/2 reading (one shared max over the whole tensor) --
    kept only for comparison against the per-channel behavior.

    `clip_percentile` (e.g. 99.9) is NOT part of the paper -- it's an optional,
    budget-neutral refinement layered on top of the confirmed per-channel scale:
    each channel's own max can still be a local outlier relative to its own bulk
    distribution (per-channel scale only removes CROSS-channel contamination,
    not a channel's own internal skew). When set, max_abs is that percentile of
    each channel's |activations| instead of the raw max -- same bit budget (r, m,
    N are untouched), same number of "important" channels, just a more robust
    scale within each one. Requires channel_dim to be set (percentile clipping a
    single global scalar isn't the point here). See run_experiments_tuned.py.

    Returns (scale, max_abs). max_abs is returned separately because eDQA needs
    it both to derive Δ_{N+m} and to store as the de-quant scale. With
    channel_dim set, both are tensors broadcastable against the full activation
    tensor (e.g. shape (1,C,1,1) for NCHW), not scalars.
    """
    if channel_dim is None:
        if clip_percentile is not None:
            raise ValueError("clip_percentile requires channel_dim to be set")
        max_abs = x.abs().max()
    elif clip_percentile is not None:
        C = x.shape[channel_dim]
        # per-channel flatten: quantile over a few hundred-thousand elements per
        # channel, well under torch.quantile's ~16M-element ceiling (unlike a
        # single flatten of the whole tensor, which can exceed it).
        flat = x.abs().movedim(channel_dim, 0).reshape(C, -1)
        q = torch.quantile(flat, clip_percentile / 100.0, dim=1)
        shape = [1] * x.dim()
        shape[channel_dim] = C
        max_abs = q.view(shape)
    else:
        reduce_dims = [d for d in range(x.dim()) if d != channel_dim]
        max_abs = x.abs().amax(dim=reduce_dims, keepdim=True)
    scale = max_abs / _denom(n_bits)
    # guard against an all-zero activation tensor (or all-zero channel)
    scale = torch.clamp(scale, min=torch.finfo(x.dtype).eps)
    return scale, max_abs


def quant_range(n_bits: int):
    """Signed integer range for n-bit symmetric quantization."""
    return -(2 ** (n_bits - 1)), (2 ** (n_bits - 1) - 1)


def quantize_direct(
    x: torch.Tensor,
    n_bits: int,
    scale: torch.Tensor | None = None,
    channel_dim: int | None = 1,
    clip_percentile: float | None = None,
):
    """Direct uniform-symmetric quantization -> integer codes.

    Eq.(1): q = Round(x / Δ_N), clamped to the signed n-bit range. `channel_dim`
    and `clip_percentile` are only used when `scale` isn't already supplied
    (see compute_scale).
    """
    if scale is None:
        scale, _ = compute_scale(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    qmin, qmax = quant_range(n_bits)
    q = torch.round(x / scale).clamp(qmin, qmax)
    return q, scale


def dequantize_direct(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """De-quantize: Δ_N · q."""
    return q * scale


def right_shift_with_error(q_more: torch.Tensor, m: int):
    """Right-shift an (N+m)-bit integer code by m bits, keeping the lost bits.

    Returns (q_shifted, se, low_bits):
      * q_shifted  : the stored N-bit code  = floor(q_more / 2^m)
      * se         : fractional shifting error  = low_bits / 2^m   (in [0, 1))
      * low_bits   : the raw lower-m-bit integer (0 .. 2^m-1), for compression

    Arithmetic (>>) and bit-mask (&) are used so this matches "reading the lower
    m bits" for both signs (activations can be negative for ViT etc.).
    """
    q_int = q_more.to(torch.int64)
    mask = (1 << m) - 1
    low_bits = q_int & mask                # 0 .. 2^m - 1
    q_shifted = q_int >> m                 # arithmetic shift == floor division
    se = low_bits.to(torch.float32) / float(2 ** m)
    return q_shifted.to(q_more.dtype), se.to(q_more.dtype), low_bits


def shifting_error_table(m: int) -> dict[int, float]:
    """Pre-computed table mapping lower-m-bit code -> decimal shifting error.

    2^m entries, e.g. m=3 -> {0:0.0, 1:0.125, ..., 7:0.875}.
    """
    return {k: k / float(2 ** m) for k in range(2 ** m)}


def fake_quantize_direct(
    x: torch.Tensor, n_bits: int, channel_dim: int | None = 1, clip_percentile: float | None = None
) -> torch.Tensor:
    """Quantize then de-quantize in one call (simulated / 'fake' quantization).

    Used by the accuracy-simulation hooks: activations are replaced by their
    quantized-then-recovered values while the network runs in floating point.
    """
    q, scale = quantize_direct(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    return dequantize_direct(q, scale)
