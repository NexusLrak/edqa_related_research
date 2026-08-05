"""
pure_energy_ranking.py
=================================================

Control for rank_energy_cascade.py's margin_threshold=3pp result (67.03% vs
the pure rank-direction surrogate's 63.82%, calib_seed=1, clip_p999, eDQA
3-bit, full test set). 20/21 layers in that run fell under the margin
threshold and got routed to energy anyway -- so it's unclear whether the win
came from the margin-gating logic, or whether energy is simply a better
importance signal than rank for this architecture regardless of margin (in
which case the cascade isn't doing anything beyond "use energy everywhere").

This ranks EVERY layer by energy (rank_surrogates.rank_channels_via_energy --
no rank surrogate, no direction search, no isolated per-layer eval at all)
and evaluates on the same seed/variant/full test set, so it's directly
comparable to both the cascade (67.03%) and the pure rank-direction surrogate
(63.82%).

Usage:
    python -m quant.diagnostics.pure_energy_ranking
"""

from __future__ import annotations

import os

import torch

from ..compression import get_compressor
from ..data import calibration_loader
from ..hooks import QuantManager
from ..rank_surrogates import rank_channels_via_energy
from ..ranking import evaluate_accuracy, load_ranks, save_ranks
from ..run_experiments import EXPERIMENTS
from ..run_experiments_tuned import VARIANTS

EXPERIMENT = "resnet18_tinyimagenet"
VARIANT = "clip_p999"
CALIB_SEED = 1
N_BITS = 3

# from the two runs already done today (2026-07-18), same seed/variant, full test set
PURE_RANK_SURROGATE = 0.6382
RANK_ENERGY_CASCADE = 0.6703


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    clip_percentile = VARIANTS[VARIANT]["clip_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    comp = get_compressor("identity")

    rank_path = f"ranks_{EXPERIMENT}_pure_energy_calibseed{CALIB_SEED}.json"
    if os.path.exists(rank_path):
        ranks = load_ranks(rank_path)
    else:
        calib = calibration_loader(train_set, cfg["calib_size"], seed=CALIB_SEED)
        ranks = rank_channels_via_energy(model, calib, layer_names, device, channel_dim=channel_dim)
        save_ranks(ranks, rank_path)

    with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
        for name in layer_names:
            if name in ranks:
                mgr.set_edqa(name, N_BITS, 3, cfg["r"], ranks[name], comp)
        acc = evaluate_accuracy(model, test_loader, device, max_batches=None)

    print("\n" + "=" * 70)
    print(f"eDQA {N_BITS}-bit, {VARIANT}, calib_seed={CALIB_SEED}, full test set")
    print("=" * 70)
    print(f"  pure rank-direction surrogate      : {PURE_RANK_SURROGATE*100:.2f}%")
    print(f"  rank+energy cascade (margin<3pp)   : {RANK_ENERGY_CASCADE*100:.2f}%")
    print(f"  pure energy (every layer)          : {acc*100:.2f}%")


if __name__ == "__main__":
    main()
