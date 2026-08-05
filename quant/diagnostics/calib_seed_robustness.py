"""
calib_seed_robustness.py
=================================================

Checks whether the headline eDQA 3-bit number (clip_p999 variant, ResNet-18/
TinyImageNet, full test set) is sensitive to WHICH calibration subset gets
drawn for ranking -- the one variable neither the greedy nor the per-layer
surrogate ranking has been tested against yet (see the 2026-07-18 discussion:
the 10K-image full test set already makes evaluation noise an unlikely
explanation for the surrogate-vs-greedy gap; calibration-subset sensitivity is
the real untested source of variance).

Scope (deliberately narrow -- see chat): ONLY the eDQA 3-bit accuracy on the
full test set, for clip_p999, greedy vs surrogate ranking, across calib_seed
in {0, 1, 2}. Not Table 2's other methods/bits, not Figure 3/4 -- those are
trend curves, less critical to re-validate right now.

Usage:
    python -m quant.diagnostics.calib_seed_robustness
"""

from __future__ import annotations

import statistics

import torch

from ..compression import get_compressor
from ..hooks import QuantManager
from ..ranking import evaluate_accuracy
from ..results_logger import RunContext
from ..run_experiments import EXPERIMENTS
from ..run_experiments_tuned import VARIANTS, _get_ranks

EXPERIMENT = "resnet18_tinyimagenet"
VARIANT = "clip_p999"
CALIB_SEEDS = (0, 1, 2)
RANKINGS = ("greedy", "surrogate")
N_BITS = 3


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    clip_percentile = VARIANTS[VARIANT]["clip_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    comp = get_compressor("identity")

    results: dict[str, list[float]] = {r: [] for r in RANKINGS}

    for ranking in RANKINGS:
        for calib_seed in CALIB_SEEDS:
            ctx = RunContext(
                experiment=EXPERIMENT, pipeline="tuned", variant=VARIANT, ranking=ranking,
                seed=0, eval_batches=None,
            )
            print(f"\n--- ranking={ranking}, calib_seed={calib_seed} ---")
            ranks = _get_ranks(
                EXPERIMENT, VARIANT, ranking, cfg, model, train_set, layer_names, channel_dim,
                clip_percentile, device, calib_seed, ctx,
            )
            with ctx.timed("edqa_3bit_fullset_eval"):
                with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=clip_percentile) as mgr:
                    for name in layer_names:
                        if name in ranks:
                            mgr.set_edqa(name, N_BITS, 3, cfg["r"], ranks[name], comp)
                    acc = evaluate_accuracy(model, test_loader, device, max_batches=None)
            print(f"  eDQA {N_BITS}-bit, full test set: {acc * 100:.2f}%")
            ctx.log_table2({"edqa": {N_BITS: (acc, 0.0)}})
            ctx.flush()
            results[ranking].append(acc)

    print("\n" + "=" * 70)
    print(f"SUMMARY: eDQA {N_BITS}-bit, {VARIANT}, full test set, across calib_seed={CALIB_SEEDS}")
    print("=" * 70)
    for ranking in RANKINGS:
        accs = results[ranking]
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {ranking:10s}: " + "  ".join(f"{a*100:.2f}%" for a in accs) +
              f"   mean={mean*100:.2f}%  std={std*100:.2f}pp")
    print(f"\n  (paper DQA(m=3) 3-bit for ResNet-18/TinyImageNet: 63.61%)")


if __name__ == "__main__":
    main()
