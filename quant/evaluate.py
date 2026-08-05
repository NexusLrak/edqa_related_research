"""
evaluate.py
=================================================

Ties the modules together to reproduce the paper's tables and figures:

  * compare_methods   -> Table 2  (Direct / PoT / NoisyQuant / eDQA over 3,4,5 bits)
  * sweep_ratio       -> Figure 3 (accuracy vs important ratio r, 10%..90%)
  * sweep_extra_bits  -> Figure 4 (accuracy vs extra bits m in {1,2,3})
  * compression_table -> Table 1 (ratio + latency per compressor)

Paper defaults: m = 3 everywhere; r = 0.55 for ResNet-18, 0.40 for all others;
randomized runs averaged over 5 repeats.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch

from .baselines import fake_quantize_noisyquant, fake_quantize_pot
from .compression import get_compressor
from .edqa_layer import edqa_fake_quantize
from .hooks import LayerConfig, QuantManager
from .quantizer import fake_quantize_direct
from .ranking import evaluate_accuracy


# --------------------------------------------------------------------------- #
# A generic manager that can drive any per-channel fake-quant function.
# --------------------------------------------------------------------------- #
class MethodManager:
    """Registers hooks that apply an arbitrary activation transform per layer."""

    def __init__(
        self,
        model,
        layer_names,
        transform: "Callable[[torch.Tensor], torch.Tensor] | dict[str, Callable[[torch.Tensor], torch.Tensor]]",
    ):
        """transform: a single callable applied to every hooked layer (the common
        case, all layers share one channel_dim), or a {layer_name: callable} dict
        for architectures that mix channel_dim conventions (e.g. ViT — see
        hooks.vit_full_target_layers / compare_methods' channel_dim handling).
        """
        self._handles = []
        modules = dict(model.named_modules())
        transforms = transform if isinstance(transform, dict) else {name: transform for name in layer_names}
        for name in layer_names:
            self._handles.append(modules[name].register_forward_hook(self._make_hook(transforms[name])))

    @staticmethod
    def _make_hook(transform: Callable[[torch.Tensor], torch.Tensor]):
        def hook(_m, _i, output):
            if isinstance(output, torch.Tensor):
                return transform(output)
            # see QuantManager._make_hook's identical handling for why: torchvision's
            # MultiheadAttention returns (attn_output, attn_weights), not a bare Tensor.
            if isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                return (transform(output[0]), *output[1:])
            return output
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def _method_transform(method: str, n_bits: int, channel_dim: int = 1, clip_percentile: float | None = None, **kw):
    if method == "direct":
        return lambda x: fake_quantize_direct(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    if method == "pot":
        # PoT's own exponent computation isn't threaded through compute_scale (it's
        # a different baseline algorithm with its own normalization), so channel_dim
        # / clip_percentile don't apply here -- left as the paper's own definition.
        return lambda x: fake_quantize_pot(x, n_bits)
    if method == "noisyquant":
        iters = kw.get("noise_search_iters", 10)
        return lambda x: fake_quantize_noisyquant(
            x, n_bits, iters, kw.get("scale_search_iters", 1), channel_dim=channel_dim,
            clip_percentile=clip_percentile,
        )
    raise ValueError(f"unknown method {method}")


@torch.no_grad()
def _avg_accuracy(model, loader, device, repeats, run_once):
    accs = [run_once() for _ in range(repeats)]
    return float(np.mean(accs)), float(np.std(accs))


def compare_methods(
    model,
    test_loader,
    layer_names,
    ranks,
    bit_levels=(3, 4, 5),
    m: int = 3,
    r: float = 0.40,
    compressor_name: str = "identity",
    device: str = "cpu",
    repeats: int = 1,
    max_batches: Optional[int] = None,
    channel_dim: "int | dict[str, int]" = 1,
    clip_percentile: float | None = None,
) -> dict:
    """Reproduce Table 2: accuracy of each method at each bit level.

    `ranks` is the {layer: [channel_ids]} table from ranking.rank_channels.
    Returns {method: {bits: (mean_acc, std_acc)}}.

    compressor_name defaults to "identity" (compression.IdentityCompressor), not
    a real codec. Every codec here is LOSSLESS, so the shifting errors decode back
    bit-exact regardless of which one is used — the choice only affects storage
    ratio / latency (Table 1), never the round-tripped values these accuracy sweeps
    measure. Huffman's encode/decode are pure-Python per-element loops; over every
    important channel of every layer of every batch that dwarfs everything else in
    the run (profiled at >90% of wall time with no effect on the accuracy number).
    Use compression_table() with a real codec when you need Table 1's numbers.

    channel_dim must match whatever `ranks` was computed with (1 for conv/linear
    NCHW-style layers, 2 for ViT's (B, N_tokens, hidden_dim) MLP-block outputs --
    see hooks.vit_mlp_target_layers). Getting this wrong silently mismatches the
    importance mask against the wrong tensor axis.
    """
    results: dict[str, dict[int, tuple[float, float]]] = {}

    def acc_now():
        return evaluate_accuracy(model, test_loader, device, max_batches)

    for method in ("direct", "pot", "noisyquant"):
        results[method] = {}
        for n in bit_levels:
            def run_once(method=method, n=n):
                if isinstance(channel_dim, dict):
                    transform = {
                        name: _method_transform(method, n, channel_dim=channel_dim[name], clip_percentile=clip_percentile)
                        for name in layer_names
                    }
                else:
                    transform = _method_transform(method, n, channel_dim=channel_dim, clip_percentile=clip_percentile)
                with MethodManager(model, layer_names, transform):
                    return acc_now()
            results[method][n] = _avg_accuracy(model, test_loader, device, repeats, run_once)

    results["edqa"] = {}
    comp = get_compressor(compressor_name)
    for n in bit_levels:
        def run_once(n=n):
            with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
                for name in layer_names:
                    if name in ranks:
                        mgr.set_edqa(name, n, m, r, ranks[name], comp)
                return acc_now()
        results["edqa"][n] = _avg_accuracy(model, test_loader, device, repeats, run_once)

    return results


def sweep_ratio(
    model,
    test_loader,
    layer_names,
    ranks,
    ratios=tuple(i / 10 for i in range(1, 10)),
    n_bits: int = 3,
    m: int = 3,
    compressor_name: str = "identity",
    device: str = "cpu",
    max_batches: Optional[int] = None,
    channel_dim: "int | dict[str, int]" = 1,
    clip_percentile: float | None = None,
) -> dict[float, float]:
    """Reproduce Figure 3: accuracy vs important-channel ratio r (m fixed, 3-bit).

    compressor_name defaults to "identity" — see compare_methods' docstring for why.
    """
    comp = get_compressor(compressor_name)
    out: dict[float, float] = {}
    for r in ratios:
        with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
            for name in layer_names:
                if name in ranks:
                    mgr.set_edqa(name, n_bits, m, r, ranks[name], comp)
            out[r] = evaluate_accuracy(model, test_loader, device, max_batches)
    return out


def sweep_extra_bits(
    model,
    test_loader,
    layer_names,
    ranks,
    ms=(1, 2, 3),
    n_bits: int = 3,
    r: float = 0.40,
    compressor_name: str = "identity",
    device: str = "cpu",
    max_batches: Optional[int] = None,
    channel_dim: "int | dict[str, int]" = 1,
    clip_percentile: float | None = None,
) -> dict[int, float]:
    """Reproduce Figure 4: accuracy vs extra bits m (r fixed, 3-bit).

    compressor_name defaults to "identity" — see compare_methods' docstring for why.
    """
    comp = get_compressor(compressor_name)
    out: dict[int, float] = {}
    for m in ms:
        with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
            for name in layer_names:
                if name in ranks:
                    mgr.set_edqa(name, n_bits, m, r, ranks[name], comp)
            out[m] = evaluate_accuracy(model, test_loader, device, max_batches)
    return out


@torch.no_grad()
def compression_table(
    model,
    loader,
    layer_names,
    ranks,
    n_bits: int = 3,
    m: int = 3,
    r: float = 0.40,
    compressors=("huffman", "deflate", "lzma", "zstd"),
    device: str = "cpu",
    max_batches: Optional[int] = 1,
) -> dict[str, dict[str, float]]:
    """Reproduce Table 1: per-compressor ratio + eDQA inference latency.

    Latency here is wall-clock over the measured batches with that compressor in
    the eDQA path. In the paper, compression runs on CPU while inference runs on
    GPU; keep that split when timing on real hardware.
    """
    out: dict[str, dict[str, float]] = {}
    for cname in compressors:
        try:
            comp = get_compressor(cname)
        except ImportError as e:
            out[cname] = {"error": str(e)}
            continue

        ratios: list[float] = []

        def tf(x):
            # measure genuine compression ratio on the important channels
            from .edqa_layer import edqa_quantize_layer
            codes, errors, _ = edqa_quantize_layer(x, n_bits, m, r, _any_rank(ranks, x), comp)
            if errors:
                orig = sum(_orig_bytes(x, cid) for cid in errors)
                packed = sum(len(b) for b in errors.values())
                if packed:
                    ratios.append(orig / packed)
            return edqa_fake_quantize(x, n_bits, m, r, _any_rank(ranks, x), comp)

        t0 = time.perf_counter()
        with MethodManager(model, layer_names, tf):
            evaluate_accuracy(model, loader, device, max_batches)
        latency = time.perf_counter() - t0

        out[cname] = {
            "compression_ratio": float(np.mean(ratios)) if ratios else float("nan"),
            "latency_s": latency,
        }
    return out


def _any_rank(ranks: dict[str, list[int]], x: torch.Tensor) -> list[int]:
    """Fallback rank when a layer-specific rank isn't threaded through the hook.

    Uses the first available rank truncated/padded to this tensor's channel count.
    In real runs, thread the correct per-layer rank (see run_experiments.py).
    """
    C = x.shape[1]
    if ranks:
        base = next(iter(ranks.values()))
        base = [c for c in base if c < C]
        base += [c for c in range(C) if c not in set(base)]
        return base
    return list(range(C))


def _orig_bytes(x: torch.Tensor, channel_id: int) -> int:
    # one uint8 per activation element in the channel (the low-bit codes)
    per_channel = x.select(1, channel_id).numel()
    return per_channel
