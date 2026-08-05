"""
rank_surrogate_full_network.py
=================================================

Full-network test of the finding from the 2026-07-14 single-layer diagnostics
(quant/experiment_logs/2026-07-14_rank_surrogate_vs_greedy*.log): a cheap
SVD-based channel-importance surrogate (stable_rank / effective_rank, used
LOW-rank-first) matched or beat Algorithm 3's expensive greedy search on an
isolated layer, at ~100x+ less compute.

That test was done on ONE layer (layer1.0.relu) with all other layers left at
full precision, and under the OLD (buggy, layer-wide) scale semantics. This
script re-tests the idea for real: rank ALL of ResNet-18's target layers with
the surrogate, plug that ranking into the full compare_methods/sweep_ratio/
sweep_extra_bits pipeline (now fixed to use the author-confirmed per-channel
scale), and compare against the greedy ranking already cached from the last
full run.

Handles one wrinkle the single-layer test didn't hit: `default_target_layers`
includes the final `fc` Linear layer, whose activations are 2D (B, C) -- no
spatial dimension, so the SVD-rank-over-space concept from rank_surrogates.py
is degenerate there (every channel would score "rank 1"). Falls back to
per-channel activation magnitude range for any 2D layer.

Usage:
    python -m quant.diagnostics.rank_surrogate_full_network
"""

from __future__ import annotations

import time

import torch

from ..compression import get_compressor
from ..data import TINYIMAGENET_CALIB_SIZE, calibration_loader, tinyimagenet_loaders
from ..evaluate import compare_methods, sweep_extra_bits, sweep_ratio
from ..hooks import default_target_layers
from ..ranking import load_ranks
from ..run_experiments import TINYIMAGENET_ROOT, build_resnet18_tinyimagenet
from ..rank_surrogates import all_channel_scores

METRIC = "stable_rank"
LOW_RANK_FIRST = True
SURROGATE_CALIB_BATCHES = 1  # kept small: activations for all 21 layers are held at once
N_BITS = 3
M = 3
R = 0.55
GREEDY_RANK_CACHE = "ranks_resnet18_tinyimagenet_3bit.json"


@torch.no_grad()
def rank_channels_via_surrogate(
    model, calib_loader, layer_names, device, metric=METRIC, low_rank_first=LOW_RANK_FIRST,
    channel_dim=1, max_batches=SURROGATE_CALIB_BATCHES,
):
    """Rank every layer's channels from a single SVD pass over calibration
    activations -- no inference-accuracy loop, unlike Algorithm 3.

    Returns {layer_name: [channel_ids most-important-first]}, same shape as
    ranking.rank_channels' output, so it's a drop-in swap for `ranks` in
    compare_methods / sweep_ratio / sweep_extra_bits.
    """
    modules = dict(model.named_modules())
    act_holders = {name: {} for name in layer_names}
    handles = []
    for name in layer_names:
        def make_hook(nm):
            def hook(_m, _i, o):
                # move to CPU immediately -- holding all 21 layers' activations
                # on GPU simultaneously risks OOM; SVD itself runs fine on CPU
                # for these (small, one-batch) matrices.
                act_holders[nm]["x"] = o.detach().cpu()
            return hook
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    acts_per_layer = {name: [] for name in layer_names}
    for i, (x, _y) in enumerate(calib_loader):
        if i >= max_batches:
            break
        model(x.to(device))
        for name in layer_names:
            acts_per_layer[name].append(act_holders[name]["x"])
    for h in handles:
        h.remove()

    ranks = {}
    degenerate_layers = []
    for name in layer_names:
        acts = torch.cat(acts_per_layer[name], dim=0)
        if channel_dim != 1:
            acts = acts.movedim(channel_dim, 1)

        if acts.dim() == 4:
            scores = all_channel_scores(acts, mode="sample_spatial")[metric]
        elif acts.dim() == 2:
            # No spatial dimension (e.g. the final `fc` Linear) -- the SVD-rank
            # concept is degenerate here (every channel is trivially "rank 1").
            # Fall back to per-channel activation range as a magnitude-based proxy.
            degenerate_layers.append(name)
            scores = acts.abs().amax(dim=0) - acts.abs().amin(dim=0)  # bigger range = more important
        else:
            raise ValueError(f"layer {name}: unexpected activation ndim {acts.dim()}")

        descending = not (low_rank_first if acts.dim() == 4 else False)
        ranks[name] = scores.argsort(descending=descending).tolist()

    if degenerate_layers:
        print(f"  (fell back to magnitude-range ranking for 2D layers: {degenerate_layers})")
    return ranks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_resnet18_tinyimagenet().to(device)
    layer_names = default_target_layers(model)
    print(f"target layers: {len(layer_names)}")

    _, test_loader, train_set = tinyimagenet_loaders(TINYIMAGENET_ROOT)
    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0)

    print(f"\nranking all layers via surrogate (metric={METRIC}, low_rank_first={LOW_RANK_FIRST})...")
    t0 = time.time()
    surrogate_ranks = rank_channels_via_surrogate(model, calib, layer_names, device)
    surrogate_dt = time.time() - t0
    print(f"surrogate ranking took {surrogate_dt:.1f}s")

    greedy_ranks = None
    try:
        greedy_ranks = load_ranks(GREEDY_RANK_CACHE)
        print(f"loaded cached greedy ranks from {GREEDY_RANK_CACHE} for comparison")
    except FileNotFoundError:
        print(f"no cached greedy ranks at {GREEDY_RANK_CACHE} -- skipping greedy comparison column")

    def run_table2(ranks, label):
        print(f"\nTable 2 ({label}):")
        table2 = compare_methods(
            model, test_loader, layer_names, ranks,
            bit_levels=(3, 4, 5), m=M, r=R, device=device, repeats=1, max_batches=8,
            channel_dim=1,
        )
        for method, by_bits in table2.items():
            row = "  ".join(f"{b}b={acc*100:5.2f}%" for b, (acc, _) in by_bits.items())
            print(f"  {method:12s} {row}")
        return table2

    if greedy_ranks is not None:
        run_table2(greedy_ranks, f"greedy Algorithm 3, cached")

    surrogate_table2 = run_table2(surrogate_ranks, f"{METRIC} surrogate, {surrogate_dt:.1f}s")

    print(f"\nFigure 3 (accuracy vs r, 3-bit, {METRIC} surrogate ranking):")
    fig3 = sweep_ratio(
        model, test_loader, layer_names, surrogate_ranks, n_bits=3, m=M, device=device,
        max_batches=8, channel_dim=1,
    )
    print("  " + "  ".join(f"r={r:.1f}:{a*100:5.2f}%" for r, a in fig3.items()))

    print(f"\nFigure 4 (accuracy vs m, 3-bit, r={R}, {METRIC} surrogate ranking):")
    fig4 = sweep_extra_bits(
        model, test_loader, layer_names, surrogate_ranks, n_bits=3, r=R, device=device,
        max_batches=8, channel_dim=1,
    )
    print("  " + "  ".join(f"m={m}:{a*100:5.2f}%" for m, a in fig4.items()))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  surrogate ranking time: {surrogate_dt:.1f}s (greedy Algorithm 3 took ~900s on this model)")
    edqa_surrogate = {b: acc for b, (acc, _) in surrogate_table2["edqa"].items()}
    print(f"  eDQA + {METRIC} surrogate: " + "  ".join(f"{b}b={a*100:.2f}%" for b, a in edqa_surrogate.items()))


if __name__ == "__main__":
    main()
