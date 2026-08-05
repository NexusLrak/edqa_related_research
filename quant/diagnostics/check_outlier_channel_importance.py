"""
check_outlier_channel_importance.py
=================================================

Tests a hypothesis raised while debugging quant/README.md pitfall #11 (3-bit
accuracy collapse caused by a layer-wide scale dominated by a few outlier
activations): does Algorithm 3's greedy importance ranking tend to rank those
same outlier channels as IMPORTANT or UNIMPORTANT?

Why it matters: the paper's Eq.(1) literally reads `∆N = |max(I)|/2^(N-1)`
where I = "the important activation channels" (a subset), while Algorithm 1/2's
pseudocode uses `|max(A_layer)|` (the whole layer, all channels). Our code
follows the Algorithm 1/2 reading. If outlier-max channels are usually ranked
UNIMPORTANT, then Eq.(1)'s reading (restrict the max to only the important
subset) would sidestep the outlier problem for eDQA specifically -- which
could be the mechanism behind the paper's eDQA-vs-Direct robustness gap at
low bit-widths that our current reproduction is missing. If outlier channels
turn out to rank IMPORTANT instead, this idea doesn't help.

Usage:
    python -m quant.diagnostics.check_outlier_channel_importance
"""

from __future__ import annotations

import time

import torch

from ..data import TINYIMAGENET_CALIB_SIZE, calibration_loader, tinyimagenet_loaders
from ..ranking import rank_channels
from ..run_experiments import TINYIMAGENET_ROOT, build_resnet18_tinyimagenet

TARGET_LAYER = "layer1.0.relu"
CHANNEL_COUNT = 64
N_BITS = 3
R = 0.55  # paper's important-channel ratio for ResNet-18


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    model = build_resnet18_tinyimagenet().to(device)
    _, _, train_set = tinyimagenet_loaders(TINYIMAGENET_ROOT, batch_size=128, num_workers=0)
    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0, batch_size=128, num_workers=0)

    # Recompute the per-channel max on a fresh calibration batch (logged reference point).
    act = {}
    model.layer1[0].relu.register_forward_hook(lambda m, i, o: act.__setitem__("x", o.detach()))
    x, y = next(iter(calib))
    x = x.to(device)
    with torch.no_grad():
        model(x)
    v = act["x"]
    per_channel_max = v.abs().amax(dim=[0, 2, 3])
    top5_outlier_channels = per_channel_max.argsort(descending=True)[:5].tolist()
    print(f"top-5 highest-max channels (candidate outliers): {top5_outlier_channels}")
    print(f"their max values: {[round(m, 4) for m in per_channel_max[top5_outlier_channels].tolist()]}")
    print(f"whole-layer max_abs: {v.abs().max().item():.4f}")

    print(f"\nranking '{TARGET_LAYER}' ({CHANNEL_COUNT} channels) via Algorithm 3 greedy search...")
    t0 = time.time()
    ranks = rank_channels(
        model,
        calib,
        [TARGET_LAYER],
        n_bits=N_BITS,
        device=device,
        channel_dim=1,
        max_batches=4,
        channel_counts={TARGET_LAYER: CHANNEL_COUNT},
        progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", flush=True),
    )
    dt = time.time() - t0
    print(f"ranking took {dt:.1f}s")

    rank_list = ranks[TARGET_LAYER]  # most important first
    print(f"\nfull rank order (most important first): {rank_list}")

    print("\noutlier channel positions in the importance ranking (0 = most important):")
    for ch in top5_outlier_channels:
        pos = rank_list.index(ch)
        pct = pos / len(rank_list) * 100
        print(f"  channel {ch}: rank position {pos}/{len(rank_list)} ({pct:.1f}th percentile)")

    k = int(round(len(rank_list) * R))
    important_set = set(rank_list[:k])
    outliers_in_important = [c for c in top5_outlier_channels if c in important_set]
    print(f"\nat r={R}: {len(outliers_in_important)}/5 outlier channels land in the 'important' set: {outliers_in_important}")

    important_idx = torch.tensor(sorted(important_set), device=device)
    max_important_only = v.index_select(1, important_idx).abs().max().item()
    max_whole_layer = v.abs().max().item()
    print(f"\nmax_abs (whole layer, current implementation)     = {max_whole_layer:.4f}")
    print(f"max_abs (important-channels-only, Eq.1 reading, r={R}) = {max_important_only:.4f}")
    print(f"ratio (important-only / whole-layer): {max_important_only / max_whole_layer:.3f}")
    print(
        "\nconclusion: "
        + (
            "outlier channels are mostly UNIMPORTANT -> restricting max to important "
            "channels would meaningfully shrink the scale, worth prototyping."
            if len(outliers_in_important) <= 2
            else "outlier channels are mostly IMPORTANT -> restricting max to important "
            "channels would NOT avoid them; this fix would not help."
        )
    )


if __name__ == "__main__":
    main()
