"""
edqa_layer.py (Algorithm 1 & 2)
=================================================

Algorithm 1 (quantize):
  * top r% important channels -> quantize with N+m bits, right-shift by m,
    store the (compressed) shifting errors.
  * the rest -> Direct N-bit quantization.
  * important point: all stored codes end up at the SAME bit length N, so no
    storage / bus-bandwidth waste; the extra info lives separately in `errors`.

Algorithm 2 (de-quantize):
  * important channels -> Δ_N · (code + decompressed_se)
  * other channels     -> Δ_N · code

`channel_dim` is 1 by default (conv NCHW; for Linear activations of shape (B, F)
the feature dim is also 1). The per-channel scale (max_abs) is returned from
quantize and must be handed back to de-quantize — at de-quant time the raw
activations are gone, so the scale has to be carried, not recomputed.
"""

from __future__ import annotations

import numpy as np
import torch

from .compression import Compressor
from .quantizer import compute_scale, dequantize_direct, quantize_direct, right_shift_with_error


def top_important_ids(rank: list[int], r: float) -> set[int]:
    """Top r% channel IDs from an importance-sorted rank list (most important first)."""
    k = int(round(len(rank) * r))
    return set(int(c) for c in rank[:k])


def _channel_mask(important: set[int], C: int, ndim: int, channel_dim: int, device) -> torch.Tensor:
    mask = torch.zeros(C, dtype=torch.bool, device=device)
    if important:
        mask[torch.tensor(sorted(important), device=device, dtype=torch.long)] = True
    shape = [1] * ndim
    shape[channel_dim] = C
    return mask.view(shape)


def edqa_quantize_layer(
    x: torch.Tensor,
    n_bits: int,
    m: int,
    r: float,
    rank: list[int],
    compressor: Compressor,
    channel_dim: int = 1,
    clip_percentile: float | None = None,
):
    """Algorithm 1 for a single layer's activations.

    Returns (codes, errors, max_abs):
      * codes   : integer tensor, same shape as x, all at N-bit length
      * errors  : dict {channel_id: compressed-bytes} for important channels only
      * max_abs : per-channel |max| tensor (shape e.g. (1,C,1,1) for NCHW),
                  needed by de-quantization -- confirmed per-channel (not
                  layer-wide) by the paper's authors, see quantizer.compute_scale.

    The quantization math (round/clamp/shift) is vectorized over the whole tensor
    for both the important and non-important paths, then combined with a channel
    mask — this used to be a per-channel Python loop over every channel (slow:
    O(C) kernel launches + CPU-GPU sync per forward pass, dominant cost on layers
    with hundreds of channels). Only the shifting-error *compression* still loops,
    over `important` channels only, since each channel gets its own Huffman/
    Deflate/etc. codebook and that step is inherently per-channel.
    """
    scale_N, max_abs = compute_scale(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    scale_Nm = torch.clamp(max_abs / (2 ** (n_bits + m - 1)), min=torch.finfo(x.dtype).eps)

    important = top_important_ids(rank, r)
    C = x.shape[channel_dim]
    qmin_more, qmax_more = -(2 ** (n_bits + m - 1)), (2 ** (n_bits + m - 1) - 1)

    q_more = torch.round(x / scale_Nm).clamp(qmin_more, qmax_more)
    q_shifted, _se, low_bits = right_shift_with_error(q_more, m)
    q_direct, _ = quantize_direct(x, n_bits, scale_N)

    mask = _channel_mask(important, C, x.dim(), channel_dim, x.device)
    codes = torch.where(mask, q_shifted, q_direct)

    errors: dict[int, bytes] = {}
    if important:
        # One GPU->CPU transfer for all important channels, not one per channel.
        # With thousands of important channels across a model's layers, a
        # per-channel .cpu() call is dominated by fixed CUDA-sync overhead, not
        # the data volume — batching it into a single transfer is what actually
        # matters for speed (the compression itself is comparatively cheap).
        idx_sorted = sorted(important)
        idx_t = torch.tensor(idx_sorted, device=x.device, dtype=torch.long)
        low_important_np = low_bits.index_select(channel_dim, idx_t).detach().cpu().numpy().astype(np.uint8)
        for i, c in enumerate(idx_sorted):
            chan_arr = np.take(low_important_np, i, axis=channel_dim).reshape(-1)
            errors[c] = compressor.encode(chan_arr)

    return codes, errors, max_abs


def edqa_dequantize_layer(
    codes: torch.Tensor,
    n_bits: int,
    m: int,
    r: float,
    rank: list[int],
    max_abs: torch.Tensor,
    errors: dict[int, bytes],
    decompressor: Compressor,
    channel_dim: int = 1,
):
    """Algorithm 2 for a single layer. Returns the de-quantized float tensor.

    `max_abs` is the per-channel tensor returned by edqa_quantize_layer (NOT a
    python float -- it must be carried through as-is, not recomputed, and not
    reduced with python's max() which only works on scalars).

    Vectorized: decompresses each important channel (inherently per-channel, and
    CPU-only — decompression itself never touches the GPU) into one CPU tensor,
    then does a SINGLE CPU->GPU transfer plus one vectorized scale multiply/add
    over the whole tensor, instead of one small transfer + kernel launch per
    channel (thousands of tiny syncs dominate the cost otherwise).
    """
    scale_N = torch.clamp(max_abs / (2 ** (n_bits - 1)), min=torch.finfo(torch.float32).eps)
    important = top_important_ids(rank, r)

    se_cpu = torch.zeros(codes.shape, dtype=torch.float32)
    for c in important:
        low = decompressor.decode(errors[c])
        chan_shape = se_cpu.select(channel_dim, c).shape
        low_t = torch.as_tensor(low, dtype=torch.float32).reshape(chan_shape)
        se_cpu.select(channel_dim, c).copy_(low_t)
    se = (se_cpu / float(2 ** m)).to(codes.device, non_blocking=True)

    return scale_N * (codes.to(se.dtype) + se)


def edqa_fake_quantize(
    x: torch.Tensor,
    n_bits: int,
    m: int,
    r: float,
    rank: list[int],
    compressor: Compressor,
    channel_dim: int = 1,
    clip_percentile: float | None = None,
) -> torch.Tensor:
    """Quantize + de-quantize in one shot (simulated eDQA for accuracy evaluation).

    This is what the activation hooks call: it round-trips activations through
    eDQA so the network runs in float while feeling eDQA's quantization error.
    """
    codes, errors, max_abs = edqa_quantize_layer(
        x, n_bits, m, r, rank, compressor, channel_dim, clip_percentile=clip_percentile
    )
    return edqa_dequantize_layer(
        codes, n_bits, m, r, rank, max_abs, errors, compressor, channel_dim
    )
