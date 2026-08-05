"""
baselines.py (Direct / PoT / NoisyQuant)
=================================================

The three methods eDQA is compared against in Table 2.

  * Direct    — plain uniform-symmetric quant/de-quant (from quantizer.py).
  * PoT       — power-of-two: 1 bit sign of value, 1 bit sign of exponent,
                remaining bits for the exponent magnitude. Values snap to the
                nearest (scaled) power of two.
  * NoisyQuant— inject noise (searched offline) before quantization, remove it
                after de-quantization. The paper's full method does an online
                search over noise + activation scale; here we implement the
                noise-bias search at a faithful-but-simplified level and flag it.

All return de-quantized float tensors so they slot straight into a fake-quant
hook the same way `fake_quantize_direct` does.
"""

from __future__ import annotations

import torch

from .quantizer import compute_scale, quant_range


def fake_quantize_pot(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Power-of-two quantization.

    Bit budget: 1 (value sign) + 1 (exponent sign) + (n_bits-2) exponent bits.
    Magnitudes are normalized by |max| then snapped to the nearest 2^e.
    """
    exp_bits = max(1, n_bits - 2)
    eps = torch.finfo(x.dtype).eps
    sign = torch.sign(x)
    absx = x.abs()
    max_abs = absx.max().clamp(min=eps)
    xn = (absx / max_abs).clamp(min=eps)               # in (0, 1]
    e = torch.round(torch.log2(xn))                    # nearest exponent (<= 0)
    e = e.clamp(min=-(2 ** exp_bits - 1), max=0)
    q = sign * torch.pow(2.0, e) * max_abs
    q = torch.where(absx <= eps, torch.zeros_like(q), q)
    return q


@torch.no_grad()
def fake_quantize_noisyquant(
    x: torch.Tensor,
    n_bits: int,
    noise_search_iters: int = 10,
    scale_search_iters: int = 1,
    channel_dim: int | None = 1,
    clip_percentile: float | None = None,
) -> torch.Tensor:
    """Simplified NoisyQuant (Liu et al., 2023).

    Faithful in spirit: a noise tensor is searched to minimize quantization error,
    added before rounding, then subtracted after de-quantization. The paper's
    accelerator config uses 10 noise-search and 10 scale-search iterations.

    NOTE (simplification): the original searches a structured bias offline over
    calibration data; here the search is a lightweight per-tensor random search so
    the module is self-contained. Swap this out for the authors' released search
    if you need to match their exact numbers.
    """
    base_scale, _ = compute_scale(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    qmin, qmax = quant_range(n_bits)

    best = None
    best_err = float("inf")
    for _ in range(max(1, scale_search_iters)):
        scale = base_scale
        for _ in range(max(1, noise_search_iters)):
            noise = (torch.rand_like(x) - 0.5) * scale        # candidate uniform noise
            q = torch.round((x + noise) / scale).clamp(qmin, qmax)
            deq = q * scale - noise
            err = (deq - x).abs().mean().item()
            if err < best_err:
                best_err, best = err, deq
    return best if best is not None else x
