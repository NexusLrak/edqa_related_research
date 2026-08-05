"""
hooks.py
=================================================

The paper quantizes ONLY activations, not weights. We do this with PyTorch
forward hooks: on each target layer's output we replace the activation tensor
with its quantized-then-de-quantized version ("fake quantization"), so the
network runs in float32 while experiencing the chosen quantizer's error.

Two modes per layer:
  * "direct" with a skip-set — Direct N-bit quantization of every channel except
    the skipped ones (left full-precision). This is what Algorithm 3 (ranking)
    needs: importance is measured by how much skipping a channel helps accuracy.
  * "edqa" — the full Algorithm 1/2 round-trip using a channel rank + ratio r.
  * "off" — passthrough (full precision).

Usage:
    mgr = QuantManager(model, target_layers, channel_dim=1)
    mgr.set_direct("layer3.0.conv2", n_bits=3, skip={5})
    mgr.set_edqa("layer3.0.conv2", n_bits=3, m=3, r=0.4, rank=[...], compressor=...)
    mgr.disable("layer3.0.conv2")
    ... run inference ...
    mgr.remove()   # unregister all hooks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .compression import Compressor
from .edqa_layer import edqa_fake_quantize
from .quantizer import fake_quantize_direct


@dataclass
class LayerConfig:
    mode: str = "off"                       # "off" | "direct" | "edqa"
    n_bits: int = 3
    m: int = 3
    r: float = 0.4
    skip: set[int] = field(default_factory=set)
    rank: Optional[list[int]] = None
    compressor: Optional[Compressor] = None


def _apply_direct_with_skip(
    x: torch.Tensor, n_bits: int, skip: set[int], channel_dim: int, clip_percentile: float | None = None
):
    """Direct-quantize every channel except `skip`, which stays full precision.

    Vectorized: quantizes the whole tensor once with a per-channel scale (see
    quantizer.compute_scale -- confirmed correct by the paper's authors, not the
    literal |max(A_layer)| in Algorithm 1/2) then blends in the skipped channels
    via a boolean mask, instead of looping per channel in Python.
    """
    q = fake_quantize_direct(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    if not skip:
        return q
    C = x.shape[channel_dim]
    keep = torch.zeros(C, dtype=torch.bool, device=x.device)
    keep[torch.tensor(sorted(skip), device=x.device, dtype=torch.long)] = True
    mask_shape = [1] * x.dim()
    mask_shape[channel_dim] = C
    keep = keep.view(mask_shape)
    return torch.where(keep, x, q)


class QuantManager:
    """Registers forward hooks on named modules and transforms their outputs."""

    def __init__(
        self,
        model: nn.Module,
        target_layers: list[str],
        channel_dim: "int | dict[str, int]" = 1,
        clip_percentile: float | None = None,
    ):
        """clip_percentile: optional, NOT part of the paper -- a manager-wide,
        budget-neutral refinement (see quantizer.compute_scale). Applies to every
        layer this manager touches, in both "direct" and "edqa" modes.

        channel_dim: a single int applied to every layer (CNNs, all NCHW), or a
        {layer_name: dim} dict for architectures that mix conventions -- e.g.
        ViT's patch-embed conv is NCHW (dim=1) but its transformer-block layers
        are (B, N_tokens, C) (dim=2). See hooks.vit_full_target_layers.
        """
        self.model = model
        self._channel_dims = (
            dict(channel_dim) if isinstance(channel_dim, dict)
            else {name: channel_dim for name in target_layers}
        )
        self.clip_percentile = clip_percentile
        self.configs: dict[str, LayerConfig] = {name: LayerConfig() for name in target_layers}
        self._handles = []
        self._modules = dict(model.named_modules())
        for name in target_layers:
            if name not in self._modules:
                raise KeyError(f"layer '{name}' not found in model.named_modules()")
            handle = self._modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    # -- configuration ------------------------------------------------------ #
    def set_direct(self, name: str, n_bits: int, skip: Optional[set[int]] = None):
        cfg = self.configs[name]
        cfg.mode, cfg.n_bits, cfg.skip = "direct", n_bits, set(skip or set())

    def set_edqa(self, name, n_bits, m, r, rank, compressor):
        cfg = self.configs[name]
        cfg.mode = "edqa"
        cfg.n_bits, cfg.m, cfg.r, cfg.rank, cfg.compressor = n_bits, m, r, rank, compressor

    def disable(self, name: str):
        self.configs[name].mode = "off"

    def disable_all(self):
        for cfg in self.configs.values():
            cfg.mode = "off"

    # -- hook body ---------------------------------------------------------- #
    def _make_hook(self, name: str):
        def transform(x: torch.Tensor) -> torch.Tensor:
            cfg = self.configs[name]
            channel_dim = self._channel_dims[name]
            if cfg.mode == "direct":
                return _apply_direct_with_skip(
                    x, cfg.n_bits, cfg.skip, channel_dim, clip_percentile=self.clip_percentile
                )
            if cfg.mode == "edqa":
                assert cfg.rank is not None and cfg.compressor is not None
                return edqa_fake_quantize(
                    x, cfg.n_bits, cfg.m, cfg.r, cfg.rank, cfg.compressor, channel_dim,
                    clip_percentile=self.clip_percentile,
                )
            return x

        def hook(_module, _inp, output):
            cfg = self.configs[name]
            if cfg.mode == "off":
                return output
            if isinstance(output, torch.Tensor):
                return transform(output)
            # torchvision's MultiheadAttention (used as ViT's self_attention target
            # layer, see vit_full_target_layers) returns (attn_output, attn_weights)
            # -- quantize the tensor element, pass the rest through untouched.
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


def default_target_layers(model: nn.Module) -> list[str]:
    """Heuristic: quantize activations produced by Conv2d / Linear modules.

    Adjust per architecture — e.g. for ViT you may prefer to hook the outputs of
    attention / MLP blocks rather than every Linear.
    """
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            names.append(name)
    return names


def vit_mlp_target_layers(model: nn.Module) -> list[str]:
    """NARROW / legacy ViT target layers: one hook per transformer block's
    MLPBlock output only.

    ⚠️ 2026-07-19 finding: this was found to under-quantize ViT relative to what
    default_target_layers does for the CNN models (every Conv2d/Linear output),
    and is the root cause of a large, previously-unexplained reproduction gap:
    with only this hook set, Direct 3-bit quantization barely degraded ViT
    accuracy at all (82.26% vs 84.63% full precision, -2.37pp) whereas the
    paper's own Table 2 shows Direct 3-bit collapsing ViT (88.31% -> 15.07%,
    -73pp). Root cause: this hook set only touches 12 of the model's 37
    nn.Linear-equivalent activation-producing layers -- it entirely skips every
    self-attention output AND the 3072-dim post-first-Linear (pre-GELU)
    intermediate activation inside each MLP block, which is specifically the
    activation the ViT-quantization literature (e.g. NoisyQuant, this paper's
    own baseline) identifies as the outlier-heavy, hard-to-quantize one. Use
    vit_full_target_layers for anything meant to be paper-comparable; this
    function is kept only for the earlier (narrower) diagnostic runs it was
    used in.

    Hooking every nn.Linear via default_target_layers isn't quite right either
    (it would double-count MLPBlock's 2 Linears as 2 separate 1D-channel layers
    instead of respecting the (B, N_tokens, hidden_dim) tensor shape, and it
    can't reach self-attention's output at all -- see vit_full_target_layers'
    docstring for why). MLPBlock's own output is a plain tensor of shape
    (B, N_tokens, hidden_dim) — hidden_dim is the natural "channel" axis.

    IMPORTANT: activations here are (B, N, C) — the channel axis is 2, not 1.
    Pass channel_dim=2 to QuantManager / rank_channels when using these layers.
    """
    names = []
    for name, module in model.named_modules():
        if name.endswith(".mlp") and not isinstance(module, nn.Linear):
            names.append(name)
    return names


def vit_full_target_layers(model: nn.Module) -> "tuple[list[str], dict[str, int]]":
    """Full-coverage ViT target layers, parity with default_target_layers(CNN)
    (every activation-producing Conv2d/Linear gets its own quantization point).

    Per encoder block (forward order): self_attention output, then the MLP's
    intermediate (post-first-Linear, pre-GELU) 3072-dim activation, then the
    MLPBlock's own final 768-dim output. Plus the patch-embedding conv and the
    final classifier head, for the same reason default_target_layers includes
    a CNN's first conv and final fc.

    self_attention is hooked as the WHOLE nn.MultiheadAttention module, not
    `self_attention.out_proj`: torchvision's fast-path attention implementation
    computes the output projection via a raw `F.linear(attn_output, out_proj_
    weight, out_proj_bias)` call inside `multi_head_attention_forward`, NOT by
    invoking `self.out_proj(...)` as a submodule -- so a forward hook on
    `out_proj` itself never fires. Hooking the parent module instead correctly
    captures the fully-formed (post-out_proj) attention output; QuantManager's
    hook already knows how to unwrap MultiheadAttention's (output, weights)
    tuple return (see _make_hook).

    Returns (layer_names, channel_dims) — channel_dims is a per-layer dict
    (conv_proj / heads.head are NCHW-or-2D, dim=1; everything inside the
    transformer stack is (B, N_tokens, C), dim=2), meant to be passed straight
    through to QuantManager / rank_channels / evaluate.* as `channel_dim`.
    """
    names: list[str] = []
    channel_dims: dict[str, int] = {}

    if hasattr(model, "conv_proj"):
        names.append("conv_proj")
        channel_dims["conv_proj"] = 1

    for block_name, _block in model.encoder.layers.named_children():
        prefix = f"encoder.layers.{block_name}"
        for suffix in ("self_attention", "mlp.0", "mlp"):
            layer = f"{prefix}.{suffix}"
            names.append(layer)
            channel_dims[layer] = 2

    if hasattr(model, "heads") and hasattr(model.heads, "head"):
        names.append("heads.head")
        channel_dims["heads.head"] = 1

    return names, channel_dims


def vit_encoder_only_target_layers(model: nn.Module) -> "tuple[list[str], dict[str, int]]":
    """EncoderBlocks-only ViT target layers (36 layers: self_attention + mlp.0 +
    mlp per block, no conv_proj / heads.head).

    Scope confirmed by the eDQA paper's authors (2026-07-21 email reply): they
    only quantized activations inside the EncoderBlocks, not the patch-embedding
    conv or the classifier head. This is narrower than vit_full_target_layers
    (38 layers), which is our own broader-coverage default used for the energy-
    ranking results elsewhere in this project.

    Use this ONLY for work that needs to match the paper's exact greedy-search
    protocol (e.g. the ViT greedy full-scan) -- it is a separate, paper-aligned
    pipeline, not a replacement for vit_full_target_layers. The energy-ranking
    results already collected under the 38-layer scheme are not affected and
    are not being re-run.
    """
    names: list[str] = []
    channel_dims: dict[str, int] = {}

    for block_name, _block in model.encoder.layers.named_children():
        prefix = f"encoder.layers.{block_name}"
        for suffix in ("self_attention", "mlp.0", "mlp"):
            layer = f"{prefix}.{suffix}"
            names.append(layer)
            channel_dims[layer] = 2

    return names, channel_dims
