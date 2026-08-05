"""
rank_energy_cascade.py
=================================================

Follow-up to rank_surrogate_per_layer_direction.py's finding (2026-07-16 run,
seed 0, baseline variant): 19/21 layers had a low-vs-high isolated-eval
accuracy margin under ~2pp, i.e. within noise of a coin flip at eval_batches=8
(n ~= 1024, single-proportion SE ~= 1.5pp) -- only conv1 had a clear margin
(~25pp). So for most layers, rank_channels_via_surrogate_per_layer's direction
pick (low-rank-first vs high-rank-first) was likely noise, not signal.

This tests a margin-based cascade (rank_surrogates.rank_channels_via_surrogate_
per_layer's new `margin_threshold` param): trust the rank-direction winner only
when its margin clears the threshold; for layers within noise of each other,
fall back to a magnitude/energy ranking (sum-of-squared per-channel
activation, descending -- unambiguous direction, no coin flip) instead.

Scope: eDQA 3-bit, clip_p999, ResNet-18/TinyImageNet, calib_seed=1 (already
run for the pure per-layer-rank surrogate in calib_seed_robustness.py: seed=1
gave 63.82%), full test set, margin_threshold=3pp (~2x the eval noise floor at
eval_batches=8).

Usage:
    python -m quant.diagnostics.rank_energy_cascade
"""

from __future__ import annotations

import os

import torch

from ..compression import get_compressor
from ..data import calibration_loader
from ..hooks import QuantManager
from ..rank_surrogates import rank_channels_via_surrogate_per_layer
from ..ranking import evaluate_accuracy, load_ranks, save_ranks
from ..run_experiments import EXPERIMENTS
from ..run_experiments_tuned import VARIANTS

EXPERIMENT = "resnet18_tinyimagenet"
VARIANT = "clip_p999"
CALIB_SEED = 1
N_BITS = 3
MARGIN_THRESHOLD = 0.03  # 3pp -- ~2x the single-proportion SE at eval_batches=8

# from calib_seed_robustness.py (2026-07-18), pure per-layer-rank surrogate, calib_seed=1
PURE_SURROGATE_SEED1 = 0.6382


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    clip_percentile = VARIANTS[VARIANT]["clip_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    comp = get_compressor("identity")

    rank_path = f"ranks_{EXPERIMENT}_surrogate_cascade_{VARIANT}_calibseed{CALIB_SEED}.json"
    if os.path.exists(rank_path):
        ranks = load_ranks(rank_path)
    else:
        calib = calibration_loader(train_set, cfg["calib_size"], seed=CALIB_SEED)
        print(f"ranking all {len(layer_names)} layers via stable_rank surrogate "
              f"(rank/energy cascade, margin_threshold={MARGIN_THRESHOLD*100:.0f}pp, "
              f"variant={VARIANT}, calib_seed={CALIB_SEED})...")
        ranks = rank_channels_via_surrogate_per_layer(
            model, calib, test_loader, layer_names, device,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
            r=cfg["r"], margin_threshold=MARGIN_THRESHOLD, verbose=True,
        )
        save_ranks(ranks, rank_path)

    with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
        for name in layer_names:
            if name in ranks:
                mgr.set_edqa(name, N_BITS, 3, cfg["r"], ranks[name], comp)
        acc = evaluate_accuracy(model, test_loader, device, max_batches=None)

    print("\n" + "=" * 70)
    print(f"eDQA {N_BITS}-bit, {VARIANT}, calib_seed={CALIB_SEED}, full test set")
    print("=" * 70)
    print(f"  pure rank-direction surrogate               : {PURE_SURROGATE_SEED1*100:.2f}%")
    print(f"  rank+energy cascade (margin<{MARGIN_THRESHOLD*100:.0f}pp->energy) : {acc*100:.2f}%")
    print(f"\n  (paper DQA(m=3) 3-bit for ResNet-18/TinyImageNet: 63.61%)")


if __name__ == "__main__":
    main()
