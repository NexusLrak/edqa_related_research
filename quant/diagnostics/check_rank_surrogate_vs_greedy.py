"""
check_rank_surrogate_vs_greedy.py
=================================================

Tests rank_surrogates' cheap SVD-based channel-importance scores as substitutes
for Algorithm 3's greedy channel-importance search (ranking.rank_channels), on
the same layer1.0.relu / ResNet-18 / TinyImageNet setup used in
check_outlier_channel_importance.py.

Algorithm 3 is O(L*C) full-model inferences -- expensive by design (see
quant/README.md pitfall #5). rank_surrogates computes ALL its metrics from a
SINGLE SVD pass per channel over a batch of calibration activations (see its
module docstring: "Trying many surrogates is therefore nearly free"): no
inference loop, no accuracy measurement at all during ranking. If any of them
correlates well with the greedy ranking's downstream accuracy, it's a much
cheaper offline-ranking substitute.

Tested metrics: numerical_rank (first tried, found degenerate -- see the
2026-07-14 log), stable_rank, effective_rank (both continuous-valued, don't
saturate at a hard threshold the way numerical_rank's rtol cutoff does).

For each metric, compares:
  1. Spearman rank correlation against the greedy Algorithm-3 ordering.
  2. Downstream eDQA accuracy (n_bits=3, m=3, r=0.55) on layer1.0.relu alone,
     using each ranking to pick the "important" channel set -- against a
     random-rank control and the no-importance Direct baseline.
  3. Wall-clock cost vs. the greedy search.

Usage:
    python -m quant.diagnostics.check_rank_surrogate_vs_greedy
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
N_BITS = 3
M = 3
R = 0.55  # paper's important-channel ratio for ResNet-18
SURROGATE_CALIB_BATCHES = 1  # deliberately cheap: 1 batch of 128 images
METRICS = ["numerical_rank", "stable_rank", "effective_rank"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)
    random.seed(0)

    model = build_resnet18_tinyimagenet().to(device)
    _, test_loader, train_set = tinyimagenet_loaders(TINYIMAGENET_ROOT, batch_size=128, num_workers=0)
    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0, batch_size=128, num_workers=0)

    # --- 0. no-importance Direct baseline, for reference ---
    with MethodManager(model, [TARGET_LAYER], _method_transform("direct", N_BITS)):
        acc_direct = evaluate_accuracy(model, test_loader, device, max_batches=8)
    print(f"reference: Direct {N_BITS}-bit, ONLY {TARGET_LAYER} (no importance ranking): {acc_direct * 100:.2f}%")

    # --- 1. greedy Algorithm 3 ranking (reference / "ground truth") ---
    print(f"\nranking '{TARGET_LAYER}' via Algorithm 3 greedy search...")
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
        greedy_importance_desc[ch] = CHANNEL_COUNT - pos  # higher = more important

    # --- 2. collect calibration activations once, correctly (see note) ---
    # NOTE: torchvision's BasicBlock reuses the SAME nn.ReLU instance twice per
    # forward() -- once after bn1 (mid-block) and once after the residual add
    # (the block's actual output). A hook that *appends* every call would silently
    # mix those two different tensors together as if they were extra samples of
    # the same thing. Use a single-slot dict (overwritten each call) so only the
    # LAST call per forward -- the block's real output, matching what
    # "layer1.0.relu" means everywhere else in this codebase -- is kept.
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
    acts = torch.cat(act_batches, dim=0)  # [N, C, H, W]

    # --- 3. all surrogate metrics from ONE svd pass (the whole point of the module) ---
    all_scores = all_channel_scores(acts, mode="sample_spatial")
    collect_dt = time.time() - t0
    print(f"\ncollected calibration activations + computed all surrogate metrics in {collect_dt:.2f}s"
          f"  ({greedy_dt / max(collect_dt, 1e-6):.0f}x faster than greedy)")

    comp = get_compressor("identity")
    random_rank = list(range(CHANNEL_COUNT))
    random.shuffle(random_rank)

    def eval_with_rank(name, rank_list):
        with QuantManager(model, [TARGET_LAYER], channel_dim=1) as mgr:
            mgr.set_edqa(TARGET_LAYER, N_BITS, M, R, rank_list, comp)
            acc = evaluate_accuracy(model, test_loader, device, max_batches=8)
        print(f"  eDQA {N_BITS}-bit (m={M}, r={R}), rank={name}: {acc * 100:.2f}%")
        return acc

    print(f"\nreference downstream accuracies on {TARGET_LAYER} alone (8 test batches):")
    acc_greedy = eval_with_rank("greedy (Algorithm 3)", greedy_rank)
    acc_random = eval_with_rank("random (control)", random_rank)

    results = {}
    for metric in METRICS:
        print(f"\n--- metric: {metric} ---")
        scores = all_scores[metric]
        print(f"raw scores per channel: {[round(s, 3) for s in scores.tolist()]}")
        n_unique = len(set(round(s, 3) for s in scores.tolist()))
        print(f"unique score values: {n_unique}/{CHANNEL_COUNT} (low = saturated/degenerate metric)")

        rho = spearman(greedy_importance_desc, scores)
        print(f"spearman(greedy, {metric}) = {rho:+.3f}")

        for direction, descending in [("high-rank-first", True), ("low-rank-first", False)]:
            surrogate_rank = scores.argsort(descending=descending).tolist()
            acc_surrogate = eval_with_rank(f"{metric} ({direction})", surrogate_rank)
            gap_closed = (acc_surrogate - acc_random) / max(acc_greedy - acc_random, 1e-9) * 100
            results[(metric, direction)] = (acc_surrogate, gap_closed)
        results[(metric, "spearman")] = rho
        results[(metric, "n_unique")] = n_unique

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Direct (no ranking):  {acc_direct * 100:.2f}%")
    print(f"  eDQA + greedy:        {acc_greedy * 100:.2f}%   ({greedy_dt:.1f}s to rank)")
    print(f"  eDQA + random:        {acc_random * 100:.2f}%   (control)")
    for metric in METRICS:
        rho = results[(metric, "spearman")]
        n_unique = results[(metric, "n_unique")]
        acc_hi, gap_hi = results[(metric, "high-rank-first")]
        acc_lo, gap_lo = results[(metric, "low-rank-first")]
        print(f"  {metric:15s} (spearman={rho:+.3f}, unique={n_unique}/{CHANNEL_COUNT}):")
        print(f"      high-rank-first: {acc_hi * 100:6.2f}%   recovers {gap_hi:.0f}% of greedy's gain over random")
        print(f"      low-rank-first:  {acc_lo * 100:6.2f}%   recovers {gap_lo:.0f}% of greedy's gain over random")
    print(f"\n  (surrogate metrics computed together in {collect_dt:.2f}s, "
          f"{greedy_dt / max(collect_dt, 1e-6):.0f}x faster than the {greedy_dt:.1f}s greedy search)")


if __name__ == "__main__":
    main()
