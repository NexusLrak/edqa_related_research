"""
rank_surrogate_per_layer_direction.py
=================================================

Follow-up to rank_surrogate_full_network.py: that script assumed ONE global
direction (low-rank-first) for every layer's stable_rank ranking, and lost to
greedy Algorithm 3 by ~10 points at 3-bit on the full 21-layer network -- even
though low-rank-first WON on an isolated single-layer test (layer1.0.relu,
2026-07-14 logs). This suggests different layers may prefer different
directions (some low-rank-first, some high-rank-first), and a single global
choice is leaving accuracy on the table.

This script determines each layer's preferred direction CHEAPLY: for each
layer, quantize ONLY that layer with eDQA (all others full precision) using
ascending vs descending stable_rank order, and keep whichever wins -- 21
layers x 2 directions x one 8-batch eval each, still tiny compared to
Algorithm 3's per-CHANNEL O(L*C) search. Then builds the final ranking from
each layer's chosen direction and re-runs the full Table 2 / Figure 3 / Figure
4 comparison against both the cached greedy ranking and the always-low-rank
surrogate result.

Usage:
    python -m quant.diagnostics.rank_surrogate_per_layer_direction
"""

from __future__ import annotations

import time

import torch

from ..compression import get_compressor
from ..data import TINYIMAGENET_CALIB_SIZE, calibration_loader, tinyimagenet_loaders
from ..evaluate import compare_methods, sweep_extra_bits, sweep_ratio
from ..hooks import QuantManager, default_target_layers
from ..ranking import evaluate_accuracy, load_ranks
from ..rank_surrogates import all_channel_scores
from ..run_experiments import TINYIMAGENET_ROOT, build_resnet18_tinyimagenet

METRIC = "stable_rank"
SURROGATE_CALIB_BATCHES = 1
N_BITS = 3
M = 3
R = 0.55
EVAL_BATCHES = 8
GREEDY_RANK_CACHE = "ranks_resnet18_tinyimagenet_3bit.json"


@torch.no_grad()
def collect_all_scores(model, calib_loader, layer_names, device, metric=METRIC, channel_dim=1):
    """Same activation collection as rank_surrogate_full_network.py, but returns
    the raw per-layer score tensors (not yet turned into an order) so both
    ascending and descending rankings can be built from them."""
    modules = dict(model.named_modules())
    act_holders = {name: {} for name in layer_names}
    handles = []
    for name in layer_names:
        def make_hook(nm):
            def hook(_m, _i, o):
                act_holders[nm]["x"] = o.detach().cpu()
            return hook
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    acts_per_layer = {name: [] for name in layer_names}
    for i, (x, _y) in enumerate(calib_loader):
        if i >= SURROGATE_CALIB_BATCHES:
            break
        model(x.to(device))
        for name in layer_names:
            acts_per_layer[name].append(act_holders[name]["x"])
    for h in handles:
        h.remove()

    scores_per_layer = {}
    degenerate_layers = []
    for name in layer_names:
        acts = torch.cat(acts_per_layer[name], dim=0)
        if channel_dim != 1:
            acts = acts.movedim(channel_dim, 1)
        if acts.dim() == 4:
            scores_per_layer[name] = (all_channel_scores(acts, mode="sample_spatial")[metric], True)
        elif acts.dim() == 2:
            degenerate_layers.append(name)
            scores_per_layer[name] = (acts.abs().amax(dim=0) - acts.abs().amin(dim=0), False)
        else:
            raise ValueError(f"layer {name}: unexpected activation ndim {acts.dim()}")
    if degenerate_layers:
        print(f"  (2D fallback -- magnitude-range ranking, direction fixed to descending: {degenerate_layers})")
    return scores_per_layer


@torch.no_grad()
def choose_direction_per_layer(model, test_loader, layer_names, scores_per_layer, device):
    """For each layer with a rank-based score, evaluate isolated-single-layer
    eDQA accuracy under ascending vs descending order and keep the winner.
    2D-fallback (magnitude) layers always use descending (bigger range = more
    important) -- no direction ambiguity there, so they're skipped.
    """
    comp = get_compressor("identity")
    chosen_ranks = {}
    report = []

    for name in layer_names:
        scores, has_direction_choice = scores_per_layer[name]
        if not has_direction_choice:
            chosen_ranks[name] = scores.argsort(descending=True).tolist()
            continue

        asc_rank = scores.argsort(descending=False).tolist()
        desc_rank = scores.argsort(descending=True).tolist()

        accs = {}
        for direction, rank_list in (("low-rank-first", asc_rank), ("high-rank-first", desc_rank)):
            with QuantManager(model, [name], channel_dim=1) as mgr:
                mgr.set_edqa(name, N_BITS, M, R, rank_list, comp)
                accs[direction] = evaluate_accuracy(model, test_loader, device, EVAL_BATCHES)

        winner = max(accs, key=accs.get)
        chosen_ranks[name] = asc_rank if winner == "low-rank-first" else desc_rank
        report.append((name, accs["low-rank-first"], accs["high-rank-first"], winner))
        print(f"  {name:30s} low={accs['low-rank-first']*100:6.2f}%  high={accs['high-rank-first']*100:6.2f}%  -> {winner}")

    return chosen_ranks, report


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18_tinyimagenet().to(device)
    layer_names = default_target_layers(model)
    print(f"target layers: {len(layer_names)}")

    _, test_loader, train_set = tinyimagenet_loaders(TINYIMAGENET_ROOT)
    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0)

    print(f"\ncollecting calibration activations + computing {METRIC} for all layers...")
    t0 = time.time()
    scores_per_layer = collect_all_scores(model, calib, layer_names, device)
    collect_dt = time.time() - t0
    print(f"collection took {collect_dt:.1f}s")

    print(f"\nchoosing direction per layer (isolated single-layer eval, {EVAL_BATCHES} test batches each)...")
    t0 = time.time()
    tuned_ranks, report = choose_direction_per_layer(model, test_loader, layer_names, scores_per_layer, device)
    direction_dt = time.time() - t0
    n_low = sum(1 for r in report if r[3] == "low-rank-first")
    n_high = sum(1 for r in report if r[3] == "high-rank-first")
    print(f"direction search took {direction_dt:.1f}s -- {n_low} layers chose low-rank-first, {n_high} chose high-rank-first")

    total_surrogate_dt = collect_dt + direction_dt
    print(f"\ntotal per-layer-tuned surrogate cost: {total_surrogate_dt:.1f}s (greedy Algorithm 3 took ~900s)")

    greedy_ranks = None
    try:
        greedy_ranks = load_ranks(GREEDY_RANK_CACHE)
    except FileNotFoundError:
        pass

    def run_table2(ranks, label):
        print(f"\nTable 2 ({label}):")
        table2 = compare_methods(
            model, test_loader, layer_names, ranks,
            bit_levels=(3, 4, 5), m=M, r=R, device=device, repeats=1, max_batches=EVAL_BATCHES,
            channel_dim=1,
        )
        for method, by_bits in table2.items():
            row = "  ".join(f"{b}b={acc*100:5.2f}%" for b, (acc, _) in by_bits.items())
            print(f"  {method:12s} {row}")
        return table2

    if greedy_ranks is not None:
        run_table2(greedy_ranks, "greedy Algorithm 3, cached, for reference")

    tuned_table2 = run_table2(tuned_ranks, f"{METRIC} surrogate, per-layer direction, {total_surrogate_dt:.1f}s")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    edqa_tuned = {b: acc for b, (acc, _) in tuned_table2["edqa"].items()}
    print(f"  eDQA + per-layer-tuned {METRIC}: " + "  ".join(f"{b}b={a*100:.2f}%" for b, a in edqa_tuned.items()))
    print(f"  ranking cost: {total_surrogate_dt:.1f}s vs greedy's ~900s")
    print(f"  direction split: {n_low}/{len(layer_names)} layers low-rank-first, {n_high}/{len(layer_names)} high-rank-first")
    print("  (compare against rank_surrogate_full_network.py's always-low-rank result: eDQA 3b=28.32% 4b=70.02% 5b=75.78%)")
    print("  (compare against greedy Algorithm 3: eDQA 3b=38.57% 4b=73.24% 5b=76.17%)")


if __name__ == "__main__":
    main()
