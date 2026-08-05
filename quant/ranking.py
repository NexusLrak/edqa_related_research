"""
ranking.py
=================================================

Rank each layer's activation channels by importance, where "important" = skipping
its quantization improves accuracy (same intuition AWQ uses for weights).

Greedy search, O(L·C):
  * walk layers in the forward direction
  * for each candidate channel in the current layer: quantize everything except
    (a) the already-fixed most-important channel of each previous layer, and
    (b) the candidate channel itself in the current layer — then measure accuracy
  * the candidate whose skip gives the highest accuracy is that layer's most
    important channel; it gets fixed and carried forward (set P)
  * sort the layer's channels by accuracy to produce the rank

⚠️ Cost: every candidate requires ONE full inference pass over the calibration
set, so this is O(L·C) inferences. Use a SMALL calibration subset (the paper uses
5000 images for CIFAR-10, 2500 for TinyImageNet) and optionally `channel_subsample`
to evaluate a subset of channels per layer when C is large.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import torch

from .hooks import QuantManager


@torch.no_grad()
def evaluate_accuracy(model, data_loader, device="cpu", max_batches: Optional[int] = None) -> float:
    """Top-1 accuracy over (a subset of) a data loader."""
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(data_loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def channel_count(model, layer_name: str, channel_dim: int = 1) -> int:
    """Best-effort channel count for a layer from its weight shape."""
    module = dict(model.named_modules())[layer_name]
    if hasattr(module, "out_channels"):       # Conv2d
        return int(module.out_channels)
    if hasattr(module, "out_features"):        # Linear
        return int(module.out_features)
    # Container module (e.g. ViT's MLPBlock, hooked as a whole by vit_mlp_target_layers):
    # its output width is the LAST nn.Linear submodule's out_features (traversal order
    # follows definition order, e.g. MLPBlock = Linear(768,3072), GELU, Dropout,
    # Linear(3072,768), Dropout -- the last Linear's out_features=768 is the block's
    # actual output channel count, not the first one's).
    last_linear_out = None
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear):
            last_linear_out = sub.out_features
    if last_linear_out is not None:
        return int(last_linear_out)
    raise ValueError(f"cannot infer channel count for '{layer_name}'")


def rank_channels(
    model,
    calib_loader,
    layer_names: list[str],
    n_bits: int,
    device: str = "cpu",
    channel_dim: "int | dict[str, int]" = 1,
    max_batches: Optional[int] = None,
    channel_subsample: Optional[int] = None,
    channel_counts: Optional[dict[str, int]] = None,
    progress: Optional[Callable[[str, int, int], None]] = None,
    checkpoint_path: Optional[str] = None,
) -> dict[str, list[int]]:
    """Greedy Algorithm 3. Returns {layer_name: [channel_ids most-important-first]}.

    Args:
      layer_names       : forward-ordered list of layers to rank.
      channel_subsample : if set, only this many evenly-spaced candidate channels
                          are evaluated per layer (speed vs completeness trade-off).
      progress          : optional callback(layer_name, done, total).
      checkpoint_path   : if set, ranks are written to this path after EACH layer
                          finishes (not just once at the end) -- if the path
                          already contains a partial result (e.g. from an
                          interrupted previous call), those layers are loaded and
                          skipped instead of re-ranked. Meant for long full-scan
                          runs (channel_subsample=None on a wide/deep model, e.g.
                          the ~16h ViT full-scan) where losing hours of progress
                          to an unrelated crash/interruption partway through
                          would be expensive. ranks[layer][0] is always the
                          layer's most-important channel (see the sort below),
                          so P can be reconstructed from a loaded partial ranks
                          dict without storing it separately.
    """
    ranks: dict[str, list[int]] = {}
    P: dict[str, int] = {}                     # most-important channel of each ranked layer

    if checkpoint_path and os.path.exists(checkpoint_path):
        ranks = load_ranks(checkpoint_path)
        for layer in layer_names:
            if layer in ranks and ranks[layer]:
                P[layer] = ranks[layer][0]
        done = [l for l in layer_names if l in ranks]
        if done:
            print(f"  resuming from checkpoint: {len(done)}/{len(layer_names)} layers already ranked")

    mgr = QuantManager(model, layer_names, channel_dim=channel_dim)
    try:
        for li, layer in enumerate(layer_names):
            if layer in ranks:
                continue  # already ranked, loaded from checkpoint
            dim = channel_dim[layer] if isinstance(channel_dim, dict) else channel_dim
            C = (channel_counts or {}).get(layer) or channel_count(model, layer, dim)
            candidates = list(range(C))
            if channel_subsample and channel_subsample < C:
                step = C / channel_subsample
                candidates = sorted({int(i * step) for i in range(channel_subsample)})

            scored: list[tuple[int, float]] = []
            for k, ch in enumerate(candidates):
                # configure previous layers: Direct quant, skip their fixed channel
                for prev in layer_names[:li]:
                    mgr.set_direct(prev, n_bits, skip={P[prev]} if prev in P else set())
                # current layer: Direct quant, skip the candidate channel
                mgr.set_direct(layer, n_bits, skip={ch})
                # later layers: full precision (not yet ranked)
                for nxt in layer_names[li + 1:]:
                    mgr.disable(nxt)

                acc = evaluate_accuracy(model, calib_loader, device, max_batches)
                scored.append((ch, acc))
                if progress:
                    progress(layer, k + 1, len(candidates))

            best_channel = max(scored, key=lambda t: t[1])[0]
            P[layer] = best_channel

            # channels not evaluated go to the bottom of the rank (least important)
            scored.sort(key=lambda t: t[1], reverse=True)
            ranked_ids = [ch for ch, _ in scored]
            evaluated = set(ranked_ids)
            ranked_ids += [c for c in range(C) if c not in evaluated]
            ranks[layer] = ranked_ids

            if checkpoint_path:
                save_ranks(ranks, checkpoint_path)
    finally:
        mgr.remove()

    return ranks


def save_ranks(ranks: dict[str, list[int]], path: str):
    import json
    with open(path, "w") as f:
        json.dump(ranks, f)


def load_ranks(path: str) -> dict[str, list[int]]:
    import json
    with open(path) as f:
        return json.load(f)
