"""
greedy_fullscan_robustness.py
=================================================

Follow-up to calib_seed_robustness.py. That run found greedy's eDQA 3-bit
accuracy IDENTICAL (0.00pp std) across calib_seed in {0,1,2} -- traced to
channel_subsample=32 (a speed shortcut, not in the paper): at r=0.55, k=
round(C*r) exceeds 32 for every one of ResNet-18's 21 target layers, so most
of each layer's "important" set is unevaluated arbitrary fill (index order),
not a real accuracy-based decision -- explaining why calibration data barely
matters to greedy's output as currently configured.

This script re-runs ONLY greedy (channel_subsample=None -- literal Algorithm
3, every channel evaluated) across the same 3 calib_seeds, to see whether the
"important" set -- and downstream eDQA 3-bit accuracy -- actually changes once
channel selection is no longer mostly arbitrary. The surrogate ranking is NOT
re-run here: it has no channel_subsample concept (rank_channels_via_surrogate_
per_layer scores every channel via a single batched SVD regardless), so
nothing about it depends on this parameter -- its numbers from
calib_seed_robustness.py (64.77% / 63.82% / 61.34%, mean=63.31%, std=1.77pp)
are reused as-is for the final comparison.

Cost: ~5000 candidates/seed vs 672 with channel_subsample=32 (~7.4x) --
expect ~100 min per seed, ~5h for all 3.

Usage:
    python -m quant.diagnostics.greedy_fullscan_robustness
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
N_BITS = 3

# from calib_seed_robustness.py (2026-07-18), NOT re-run here -- see module docstring
SURROGATE_RESULTS = {0: 0.6477, 1: 0.6382, 2: 0.6134}
GREEDY_SUBSAMPLED_RESULTS = {0: 0.6320, 1: 0.6320, 2: 0.6320}


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    clip_percentile = VARIANTS[VARIANT]["clip_percentile"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    comp = get_compressor("identity")

    fullscan_results: list[float] = []

    for calib_seed in CALIB_SEEDS:
        ctx = RunContext(
            experiment=EXPERIMENT, pipeline="tuned", variant=VARIANT, ranking="greedy_fullscan",
            seed=0, eval_batches=None,
        )
        print(f"\n--- ranking=greedy_fullscan (channel_subsample=None), calib_seed={calib_seed} ---")
        ranks = _get_ranks(
            EXPERIMENT, VARIANT, "greedy", cfg, model, train_set, layer_names, channel_dim,
            clip_percentile, device, calib_seed, ctx, channel_subsample=None,
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
        fullscan_results.append(acc)

    mean_fs = statistics.mean(fullscan_results)
    std_fs = statistics.stdev(fullscan_results) if len(fullscan_results) > 1 else 0.0

    print("\n" + "=" * 70)
    print(f"SUMMARY: eDQA {N_BITS}-bit, {VARIANT}, full test set, calib_seed={CALIB_SEEDS}")
    print("=" * 70)
    gs = [GREEDY_SUBSAMPLED_RESULTS[s] for s in CALIB_SEEDS]
    sg = [SURROGATE_RESULTS[s] for s in CALIB_SEEDS]
    print(f"  greedy (subsample=32)  : " + "  ".join(f"{a*100:.2f}%" for a in gs) +
          f"   mean={statistics.mean(gs)*100:.2f}%  std={(statistics.stdev(gs) if len(gs)>1 else 0)*100:.2f}pp")
    print(f"  greedy (fullscan)      : " + "  ".join(f"{a*100:.2f}%" for a in fullscan_results) +
          f"   mean={mean_fs*100:.2f}%  std={std_fs*100:.2f}pp")
    print(f"  surrogate (per-layer)  : " + "  ".join(f"{a*100:.2f}%" for a in sg) +
          f"   mean={statistics.mean(sg)*100:.2f}%  std={(statistics.stdev(sg) if len(sg)>1 else 0)*100:.2f}pp")
    print(f"\n  (paper DQA(m=3) 3-bit for ResNet-18/TinyImageNet: 63.61%)")


if __name__ == "__main__":
    main()
