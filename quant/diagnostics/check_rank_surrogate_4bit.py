"""
check_rank_surrogate_4bit.py
=================================================

Follow-up to check_rank_surrogate_vs_greedy.py (see the 2026-07-14 log there):
at 3-bit, stable_rank and effective_rank -- used LOW-rank-first (a channel with
a more concentrated/structured response across calibration samples is treated
as more important, the reverse of the "rank = diversity = importance" HRank-style
intuition) -- both beat the expensive greedy Algorithm 3 search on layer1.0.relu,
while costing ~1/120th the compute.

This script repeats the comparison at 4-bit to check whether that holds at a
different bit-width, restricted to what's already been validated as promising:
greedy vs. random vs. {stable_rank, effective_rank} in low-rank-first order only
(numerical_rank was dropped -- degenerate/saturated at this layer; high-rank-first
was dropped -- already shown worse than random for both metrics).

Usage:
    python -m quant.diagnostics.check_rank_surrogate_4bit
"""

from __future__ import annotations

import random
import time

import torch

from ..compression import get_compressor
from ..data import TINYIMAGENET_CALIB_SIZE, calibration_loader, tinyimagenet_loaders
from ..evaluate import MethodManager, _method_transform
from ..hooks import QuantManager
from ..rank_surrogates import all_channel_scores, spearman
from ..ranking import evaluate_accuracy, rank_channels
from ..run_experiments import TINYIMAGENET_ROOT, build_resnet18_tinyimagenet

TARGET_LAYER = "layer1.0.relu"
CHANNEL_COUNT = 64
N_BITS = 4
M = 3
R = 0.55  # paper's important-channel ratio for ResNet-18
SURROGATE_CALIB_BATCHES = 1
METRICS = ["stable_rank", "effective_rank"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)
    random.seed(0)

    model = build_resnet18_tinyimagenet().to(device)
    _, test_loader, train_set = tinyimagenet_loaders(TINYIMAGENET_ROOT, batch_size=128, num_workers=0)
    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0, batch_size=128, num_workers=0)

    with MethodManager(model, [TARGET_LAYER], _method_transform("direct", N_BITS)):
        acc_direct = evaluate_accuracy(model, test_loader, device, max_batches=8)
    print(f"reference: Direct {N_BITS}-bit, ONLY {TARGET_LAYER} (no importance ranking): {acc_direct * 100:.2f}%")

    print(f"\nranking '{TARGET_LAYER}' via Algorithm 3 greedy search (n_bits={N_BITS})...")
    t0 = time.time()
    ranks = rank_channels(
        model, calib, [TARGET_LAYER], n_bits=N_BITS, device=device,
        channel_dim=1, max_batches=4, channel_counts={TARGET_LAYER: CHANNEL_COUNT},
        progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", flush=True),
    )
    greedy_dt = time.time() - t0
    greedy_rank = ranks[TARGET_LAYER]
    print(f"greedy ranking took {greedy_dt:.1f}s")
    print(f"greedy rank order (most important first): {greedy_rank}")
    greedy_importance_desc = torch.zeros(CHANNEL_COUNT)
    for pos, ch in enumerate(greedy_rank):
        greedy_importance_desc[ch] = CHANNEL_COUNT - pos

    # same activation-collection fix as check_rank_surrogate_vs_greedy.py: the
    # BasicBlock's nn.ReLU is called twice per forward (mid-block + block output),
    # so only keep the LAST call per forward pass (a single-slot dict, not a list).
    act_holder: dict = {}
    hook_handle = model.layer1[0].relu.register_forward_hook(
        lambda m, i, o: act_holder.__setitem__("x", o.detach())
    )
    t0 = time.time()
    act_batches = []
    with torch.no_grad():
        for i, (x, y) in enumerate(calib):
            if i >= SURROGATE_CALIB_BATCHES:
                break
            model(x.to(device))
            act_batches.append(act_holder["x"])
    hook_handle.remove()
    acts = torch.cat(act_batches, dim=0)

    all_scores = all_channel_scores(acts, mode="sample_spatial")
    collect_dt = time.time() - t0
    speedup = greedy_dt / max(collect_dt, 1e-6)
    print(f"\ncollected calibration activations + computed surrogate metrics in {collect_dt:.2f}s"
          f"  ({speedup:.0f}x faster than greedy)")

    comp = get_compressor("identity")
    random_rank = list(range(CHANNEL_COUNT))
    random.shuffle(random_rank)

    def eval_with_rank(name, rank_list):
        with QuantManager(model, [TARGET_LAYER], channel_dim=1) as mgr:
            mgr.set_edqa(TARGET_LAYER, N_BITS, M, R, rank_list, comp)
            acc = evaluate_accuracy(model, test_loader, device, max_batches=8)
        print(f"  eDQA {N_BITS}-bit (m={M}, r={R}), rank={name}: {acc * 100:.2f}%")
        return acc

    print(f"\ndownstream eDQA accuracy on {TARGET_LAYER} alone (8 test batches), n_bits={N_BITS}:")
    acc_greedy = eval_with_rank("greedy (Algorithm 3)", greedy_rank)
    acc_random = eval_with_rank("random (control)", random_rank)

    results = {}
    for metric in METRICS:
        scores = all_scores[metric]
        rho = spearman(greedy_importance_desc, scores)
        surrogate_rank = scores.argsort(descending=False).tolist()  # low-rank-first
        acc_surrogate = eval_with_rank(f"{metric} (low-rank-first)", surrogate_rank)
        gap_closed = (acc_surrogate - acc_random) / max(acc_greedy - acc_random, 1e-9) * 100
        results[metric] = (rho, acc_surrogate, gap_closed)

    print("\n" + "=" * 70)
    print(f"SUMMARY (n_bits={N_BITS})")
    print("=" * 70)
    print(f"  Direct (no ranking):  {acc_direct * 100:.2f}%")
    print(f"  eDQA + greedy:        {acc_greedy * 100:.2f}%   ({greedy_dt:.1f}s to rank)")
    print(f"  eDQA + random:        {acc_random * 100:.2f}%   (control)")
    for metric, (rho, acc_surrogate, gap_closed) in results.items():
        print(f"  eDQA + {metric:15s} (low-rank-first): {acc_surrogate * 100:6.2f}%   "
              f"spearman={rho:+.3f}  recovers {gap_closed:.0f}% of greedy's gain over random")
    print(f"\n  (surrogate metrics computed in {collect_dt:.2f}s, {speedup:.0f}x faster than the "
          f"{greedy_dt:.1f}s greedy search)")


if __name__ == "__main__":
    main()
