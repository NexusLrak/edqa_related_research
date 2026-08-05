"""
rank_energy_multiseed.py
=================================================

Multi-seed validation of the 2026-07-18 calib_seed=1 finding (rank_energy_
cascade.py, pure_energy_ranking.py): at seed=1, pure energy (66.70%) and the
rank+energy cascade (67.03%, margin_threshold=3pp) both meaningfully beat the
pure per-layer-rank-direction surrogate (63.82%) and the paper's own number
(63.61%) -- and the two are within 0.33pp of each other, meaning the margin
gating barely matters; energy alone does almost all the work.

One seed proves nothing on its own (see this project's whole history of
single-run claims needing correction: the 8-batch-vs-full-set gap, the
channel_subsample degeneracy). This runs BOTH pure energy and the cascade at
calib_seed in {0, 2} (seed 1 already done, hardcoded below) and prints a full
3-seed x 3-method comparison table.

Scope: eDQA 3-bit, clip_p999, ResNet-18/TinyImageNet, full test set.

Usage:
    python -m quant.diagnostics.rank_energy_multiseed
"""

from __future__ import annotations

import os
import statistics

import torch

from ..compression import get_compressor
from ..data import calibration_loader
from ..hooks import QuantManager
from ..rank_surrogates import rank_channels_via_energy, rank_channels_via_surrogate_per_layer
from ..ranking import evaluate_accuracy, load_ranks, save_ranks
from ..run_experiments import EXPERIMENTS
from ..run_experiments_tuned import VARIANTS

EXPERIMENT = "resnet18_tinyimagenet"
VARIANT = "clip_p999"
CALIB_SEEDS = (0, 2)          # seed 1 already run 2026-07-18, hardcoded below
MARGIN_THRESHOLD = 0.03
N_BITS = 3

# from calib_seed_robustness.py / rank_energy_cascade.py / pure_energy_ranking.py (2026-07-18)
PURE_RANK_RESULTS = {0: 0.6477, 1: 0.6382, 2: 0.6134}
CASCADE_RESULTS = {1: 0.6703}
PURE_ENERGY_RESULTS = {1: 0.6670}


def _eval_with_ranks(model, test_loader, layer_names, channel_dim, clip_percentile, ranks, cfg):
    comp = get_compressor("identity")
    with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
        for name in layer_names:
            if name in ranks:
                mgr.set_edqa(name, N_BITS, 3, cfg["r"], ranks[name], comp)
        return evaluate_accuracy(model, test_loader, device=next(model.parameters()).device, max_batches=None)


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    clip_percentile = VARIANTS[VARIANT]["clip_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]

    for calib_seed in CALIB_SEEDS:
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)

        # -- pure energy: cheap, no isolated evals --
        energy_path = f"ranks_{EXPERIMENT}_pure_energy_calibseed{calib_seed}.json"
        if os.path.exists(energy_path):
            energy_ranks = load_ranks(energy_path)
        else:
            energy_ranks = rank_channels_via_energy(model, calib, layer_names, device, channel_dim=channel_dim)
            save_ranks(energy_ranks, energy_path)
        acc_energy = _eval_with_ranks(model, test_loader, layer_names, channel_dim, clip_percentile, energy_ranks, cfg)
        PURE_ENERGY_RESULTS[calib_seed] = acc_energy
        print(f"[seed={calib_seed}] pure energy: {acc_energy*100:.2f}%")

        # -- rank+energy cascade: needs its own isolated-eval direction search --
        cascade_path = f"ranks_{EXPERIMENT}_surrogate_cascade_{VARIANT}_calibseed{calib_seed}.json"
        if os.path.exists(cascade_path):
            cascade_ranks = load_ranks(cascade_path)
        else:
            calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
            print(f"[seed={calib_seed}] ranking via rank+energy cascade (margin_threshold={MARGIN_THRESHOLD*100:.0f}pp)...")
            cascade_ranks = rank_channels_via_surrogate_per_layer(
                model, calib, test_loader, layer_names, device,
                channel_dim=channel_dim, clip_percentile=clip_percentile,
                r=cfg["r"], margin_threshold=MARGIN_THRESHOLD, verbose=True,
            )
            save_ranks(cascade_ranks, cascade_path)
        acc_cascade = _eval_with_ranks(model, test_loader, layer_names, channel_dim, clip_percentile, cascade_ranks, cfg)
        CASCADE_RESULTS[calib_seed] = acc_cascade
        print(f"[seed={calib_seed}] rank+energy cascade: {acc_cascade*100:.2f}%")

    seeds_sorted = sorted(PURE_RANK_RESULTS)
    print("\n" + "=" * 78)
    print(f"SUMMARY: eDQA {N_BITS}-bit, {VARIANT}, full test set, calib_seed in {seeds_sorted}")
    print("=" * 78)

    def row(label, results):
        vals = [results[s] for s in seeds_sorted]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {label:32s}: " + "  ".join(f"{v*100:.2f}%" for v in vals) +
              f"   mean={mean*100:.2f}%  std={std*100:.2f}pp")

    row("pure rank-direction surrogate", PURE_RANK_RESULTS)
    row("rank+energy cascade (margin<3pp)", CASCADE_RESULTS)
    row("pure energy (every layer)", PURE_ENERGY_RESULTS)
    print(f"\n  (paper DQA(m=3) 3-bit for ResNet-18/TinyImageNet: 63.61%)")
    print(f"  (for reference, full-scan greedy: 63.95% / 65.02% / 63.90%, mean=64.29%, std=0.63pp)")


if __name__ == "__main__":
    main()
