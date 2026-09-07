"""
verify_table_4_4.py — one-off script to fill in Table 4.4's asterisked cells
with real measurements instead of interpolated/derived numbers.

IMPORTANT (protection_gain-based ranking depends on m): unlike energy/saliency,
eDQA-saliency's ranking itself is a function of m (protection_gain =
distortion_N - distortion_{N+m}), so the ranking must be recomputed for each m
in the sweep -- NOT computed once and reused downstream like
evaluate.sweep_extra_bits does for magnitude-based criteria. This script does
a proper per-m recompute.

Usage:
    python -m quant.verify_table_4_4 --experiment mobilenetv2_cifar10
    python -m quant.verify_table_4_4 --experiment vit_b16_tinyimagenet
"""
import argparse
import json
import time

import torch

from quant.compression import get_compressor
from quant.data import calibration_loader
from quant.hooks import QuantManager
from quant.rank_gradient_surrogates import rank_channels_via_edqa_saliency
from quant.ranking import evaluate_accuracy
from quant.run_experiments import EXPERIMENTS


def main(name: str, calib_seed: int = 0, clip_percentile: float = 99.9):
    cfg = EXPERIMENTS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model = cfg["build"]().to(device)
    train_loader, test_loader, train_set = cfg["loaders"]()

    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    if isinstance(layer_names, tuple):
        layer_names, channel_dim = layer_names
    print(f"{len(layer_names)} target layers, channel_dim={channel_dim}, r={cfg['r']}")

    calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)

    results = {}
    for m in (1, 2, 3):
        t0 = time.time()
        ranks = rank_channels_via_edqa_saliency(
            model, calib, layer_names, channel_dim=channel_dim, device=device,
            n_bits=3, m=m, clip_percentile=clip_percentile,
        )
        rank_seconds = time.time() - t0

        comp = get_compressor("identity")
        with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
            for ln in layer_names:
                if ln in ranks:
                    mgr.set_edqa(ln, 3, m, cfg["r"], ranks[ln], comp)
            t0 = time.time()
            acc = evaluate_accuracy(model, test_loader, device=device, max_batches=None)
            eval_seconds = time.time() - t0

        budget = 3 + cfg["r"] * m
        print(f"m={m} (budget={budget:.2f}): acc={acc*100:.2f}%  "
              f"(rank {rank_seconds:.1f}s, eval {eval_seconds:.1f}s)")
        results[m] = dict(accuracy=acc, budget=budget, rank_seconds=rank_seconds, eval_seconds=eval_seconds)

    out_path = f"quant/experiment_logs/table44_verify_{name}.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": name, "ranking": "edqa_saliency", "n_bits": 3,
                    "r": cfg["r"], "calib_seed": calib_seed, "results": results}, f, indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, choices=list(EXPERIMENTS))
    ap.add_argument("--calib-seed", type=int, default=0)
    args = ap.parse_args()
    main(args.experiment, calib_seed=args.calib_seed)
