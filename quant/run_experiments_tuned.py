"""
run_experiments_tuned.py
=================================================

Separate pipeline for independent, beyond-paper accuracy tuning -- kept apart
from run_experiments.py, which stays a faithful reproduction of the paper's
Table 2 / Figure 3 / Figure 4 (as clarified by the authors: per-channel scale,
see quantizer.compute_scale's docstring). This file is where we deliberately
deviate to chase higher accuracy, at the SAME resource budget as the paper
(same r, same m, same N) -- see quant/README.md and the 2026-07 discussion
with the advisor: bumping r or m to "buy" accuracy isn't a fair comparison
against the paper's reported numbers, since it changes how much quantization
budget eDQA is actually using. Only budget-neutral refinements belong here.

Three independent, orthogonal axes, each a plain string/int so new options are
cheap to add:

  --variant (scale computation):
    * "baseline"  : clip_percentile=None -- identical to run_experiments.py,
                    included so this pipeline can self-check against the
                    untouched reference number.
    * "clip_p999" : per-channel scale clipped to each channel's own 99.9th
                    percentile instead of its raw max (quantizer.compute_scale's
                    clip_percentile param). Per-channel scale already removes
                    CROSS-channel contamination; this additionally guards
                    against a channel's own internal outliers. Same r, m, N --
                    purely a more robust scale computation.
    * "clip_p99"  : same idea, more aggressive (99th percentile).

  --ranking (channel importance source):
    * "greedy"    : Algorithm 3's O(L*C) greedy search (ranking.rank_channels),
                    cached at ranks_{experiment}_3bit.json (shared across
                    --variant since ranking doesn't touch the scale formula).
    * "surrogate" : cheap O(L) SVD-based ranking with a per-layer-tuned
                    ascending/descending choice (rank_surrogates.
                    rank_channels_via_surrogate_per_layer) -- see
                    quant/experiment_logs/2026-07-16_rank_surrogate_*.log for
                    where this came from. Cached per (variant, calib_seed) --
                    the direction search depends on both.
    * "energy"    : per-channel sum-of-squared activation, descending
                    (rank_surrogates.rank_channels_via_energy) -- no SVD, no
                    direction search, no isolated per-layer evals, just one
                    calibration forward pass. Found 2026-07-18 (see
                    quant/experiment_logs/2026-07-18_rank_energy_multiseed.log)
                    to beat BOTH "surrogate" (mean 63.31%, std 1.77pp) and
                    full-scan "greedy" (mean 64.29%, std 0.63pp) on ResNet-18/
                    TinyImageNet eDQA 3-bit: mean 66.51%, std 0.20pp, across
                    calib_seed in {0,1,2} -- also tried gating it behind the
                    rank surrogate's per-layer direction margin (a "cascade");
                    that added nothing (mean 66.74%, std 0.30pp, statistically
                    the same modulo GPU float non-determinism) since the
                    margin threshold routed ~all layers to energy anyway, so
                    plain energy is what's implemented, not the cascade.
                    NOT yet validated on ViT-b-16 (2D MLP activations, no
                    spatial dim -- would need its own magnitude fallback,
                    see rank_surrogates.rank_channels_via_surrogate_per_layer's
                    2D branch) or ResNet-32/CIFAR-10.

  --calib-seed  : which calibration subset (calibration_loader's `seed` arg)
                  is used both for greedy ranking and the surrogate's direction
                  search. Vary this across repeats to check whether a result is
                  robust to which calibration images happened to get sampled,
                  vs an artifact of one particular subset.

IMPORTANT (2026-07 finding): an 8-batch (1024-image) test subset can read
meaningfully HIGHER than the full test set for the same config (69.34% vs
63.20% for clip_p999+greedy on ResNet-18/TinyImageNet) -- always check
`--max-batches` / the eval_batches column in results.csv before comparing
numbers across runs. Default here is still 8 for fast iteration; pass
--max-batches 0 for the full set when the number needs to be trustworthy.

Usage:
    python -m quant.run_experiments_tuned --experiment resnet18_tinyimagenet --variant clip_p999 --ranking greedy --max-batches 0
    python -m quant.run_experiments_tuned --experiment resnet18_tinyimagenet --variant clip_p999 --ranking surrogate --max-batches 0
    python -m quant.run_experiments_tuned --experiment resnet18_tinyimagenet --variant clip_p999 --ranking energy --max-batches 0
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from .data import calibration_loader
from .evaluate import compare_methods, sweep_extra_bits, sweep_ratio
from .rank_surrogates import (
    rank_channels_via_energy,
    rank_channels_via_stratified,
    rank_channels_via_surrogate_per_layer,
)
from .rank_gradient_surrogates import (
    protection_gain_scores,
    rank_channels_via_edqa_saliency,
    rank_channels_via_fisher,
    rank_channels_via_fisher_a2,
    rank_channels_via_pg2,
    rank_channels_via_saliency,
)
from .ranking import load_ranks, rank_channels, save_ranks
from .results_logger import RunContext
from .run_experiments import EXPERIMENTS

VARIANTS = {
    "baseline": dict(clip_percentile=None),
    "clip_p999": dict(clip_percentile=99.9),
    "clip_p99": dict(clip_percentile=99.0),
}

RANKINGS = (
    "greedy", "surrogate", "energy", "stratified",
    "saliency", "fisher", "fisher_a2", "protection_gain", "pg2", "edqa_saliency",
)


def _rank_cache_suffix(calib_seed: int) -> str:
    return "" if calib_seed == 0 else f"_calibseed{calib_seed}"


def _get_ranks(
    name, variant, ranking, cfg, model, train_set, layer_names, channel_dim, clip_percentile, device, calib_seed, ctx,
    channel_subsample=None,
):
    """channel_subsample only affects "greedy" (Algorithm 3 evaluates this many
    evenly-spaced candidate channels per layer instead of all C -- an optional
    speed shortcut, not part of the paper). Defaults to None: the literal,
    unabridged Algorithm 3 (every channel evaluated) -- slower, but never
    degenerate. Pass an explicit int (e.g. 32) to opt into the shortcut,
    knowingly trading accuracy fidelity for speed.

    2026-07-26: this used to default to 32, discovered (2026-07-26, while
    writing up the dissertation) to be silently degenerate whenever
    k=round(C*r) exceeds the subsample count for a layer -- confirmed on
    ResNet-18/TinyImageNet at r=0.55, where k=round(C*r) exceeded 32 for EVERY
    layer, so the top-k "important" set ended up as {all 32 evaluated
    candidates} + {arbitrary unevaluated fill, chosen by index order, never
    accuracy-tested}, and the ResNet-18 "greedy" number reported for a while
    (63.20%) was actually this degenerate shortcut, not real Algorithm 3
    (real fullscan: 64.29% mean, std=0.63pp, see RESULTS.md). Switched the
    default to None so a bare call is never silently degenerate again; the
    speed shortcut is now opt-in only.
    "surrogate" has no equivalent parameter -- rank_channels_via_surrogate_per_layer
    scores every channel in one batched SVD regardless, see rank_surrogates.py.
    """
    suffix = _rank_cache_suffix(calib_seed)

    if ranking == "greedy":
        # Shared across variants: Algorithm 3 never touches compute_scale.
        # channel_subsample=None (the default) is real fullscan -- suffix is
        # a marker for the explicit-shortcut case, not the default case.
        subsample_suffix = "" if channel_subsample is None else f"_subsample{channel_subsample}"
        rank_path = f"ranks_{name}_3bit{suffix}{subsample_suffix}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        with ctx.timed("ranking"):
            ranks = rank_channels(
                model, calib, layer_names, n_bits=3, device=device, channel_dim=channel_dim,
                channel_subsample=channel_subsample, max_batches=4,
                progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", end="\r"),
            )
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "surrogate":
        # Per-(variant, calib_seed) cache: the per-layer direction search is
        # evaluated under this variant's clip_percentile and this calibration
        # subset, so either changing could pick different directions.
        rank_path = f"ranks_{name}_surrogate_{variant}{suffix}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        _, test_loader_for_ranking, _ = cfg["loaders"]()
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        print(f"  ranking all {len(layer_names)} layers via stable_rank surrogate "
              f"(per-layer direction search, variant={variant}, calib_seed={calib_seed})...")
        with ctx.timed("ranking"):
            ranks = rank_channels_via_surrogate_per_layer(
                model, calib, test_loader_for_ranking, layer_names, device,
                channel_dim=channel_dim, clip_percentile=clip_percentile,
                r=cfg["r"], verbose=True,
            )
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "energy":
        # Cache keyed only on calib_seed -- energy is raw per-channel
        # sum-of-squared activation from calibration data, no clip_percentile
        # or scale computation involved, so it's shared across --variant
        # (unlike "surrogate", whose direction search does depend on variant).
        rank_path = f"ranks_{name}_pure_energy_calibseed{calib_seed}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        with ctx.timed("ranking"):
            ranks = rank_channels_via_energy(model, calib, layer_names, device, channel_dim=channel_dim)
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "stratified":
        # Cache keyed only on calib_seed -- same reasoning as "energy": no clip_percentile
        # dependence (rank_channels_via_stratified only reads raw activations).
        rank_path = f"ranks_{name}_stratified_calibseed{calib_seed}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        with ctx.timed("ranking"):
            ranks = rank_channels_via_stratified(
                model, calib, layer_names, device, r=cfg["r"], channel_dim=channel_dim,
            )
        save_ranks(ranks, rank_path)
        return ranks

    if ranking in ("saliency", "fisher", "fisher_a2"):
        # Cache keyed only on calib_seed -- same reasoning as "energy": these are
        # raw activation/gradient statistics from calibration data, no
        # clip_percentile/scale computation involved (see
        # rank_gradient_surrogates.compute_pq_scores).
        rank_path = f"ranks_{name}_{ranking}_calibseed{calib_seed}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        fn = {"saliency": rank_channels_via_saliency, "fisher": rank_channels_via_fisher,
              "fisher_a2": rank_channels_via_fisher_a2}[ranking]
        with ctx.timed("ranking"):
            ranks = fn(model, calib, layer_names, device, channel_dim=channel_dim, calib_batches=1)
        save_ranks(ranks, rank_path)
        return ranks

    if ranking in ("protection_gain", "pg2", "edqa_saliency"):
        # Cache keyed on (variant, calib_seed) like "surrogate": pg is computed
        # via fake_quantize_direct(..., clip_percentile=...), so it DOES depend
        # on --variant, unlike saliency/fisher/fisher_a2/energy above.
        # Reference configuration throughout this project (n=3, m=3) -- see
        # dissertation §3.2.3, "established at one reference bit-budget
        # configuration".
        rank_path = f"ranks_{name}_{ranking}_{variant}{suffix}.json"
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        with ctx.timed("ranking"):
            if ranking == "edqa_saliency":
                ranks = rank_channels_via_edqa_saliency(
                    model, calib, layer_names, channel_dim=channel_dim, device=device,
                    n_bits=3, m=3, clip_percentile=clip_percentile,
                )
            else:
                pg = protection_gain_scores(
                    model, calib, layer_names, channel_dim=channel_dim, device=device,
                    n_bits=3, m=3, clip_percentile=clip_percentile,
                )
                ranks = (
                    {n: v.argsort(descending=True).tolist() for n, v in pg.items()}
                    if ranking == "protection_gain" else rank_channels_via_pg2(pg)
                )
        save_ranks(ranks, rank_path)
        return ranks

    raise ValueError(f"unknown ranking {ranking!r}, choose from {RANKINGS}")


def run(
    name: str, variant: str, ranking: str, max_batches: "int | None" = 8, seed: int = 0, calib_seed: int = 0,
    channel_subsample: "int | None" = None,
):
    cfg = EXPERIMENTS[name]
    clip_percentile = VARIANTS[variant]["clip_percentile"]
    ctx = RunContext(
        experiment=name, pipeline="tuned", variant=variant, ranking=ranking, seed=seed, eval_batches=max_batches
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    # compare_methods/sweep_ratio/sweep_extra_bits below iterate test_loader FRESH many
    # times (Table2: 12, Fig3: 9, Fig4: 3 -- 24 total), the same repeated-fresh-iteration
    # pattern data.calibration_loader was fixed for (see its docstring): with
    # num_workers>0 (loaders() factories default to 4, tuned for the ONE long iteration
    # a training epoch does) Windows respawns worker processes on every one of those 24
    # iterations, which dwarfs the actual forward-pass cost when max_batches caps each
    # eval to a small subset. Only rebuild with num_workers=0 in that small-subset case;
    # when max_batches is None (full test set), each of the 24 iterations does enough
    # real work (the whole val set) that the one-time respawn cost is negligible next to
    # it, and losing the workers' parallel decode/resize would cost far more than it
    # saves (measured 2026-07-19: single-threaded TinyImageNet decode/resize was the
    # actual bottleneck for an 8-batch ViT eval, GPU sitting near 0% util waiting on it).
    if max_batches is not None:
        test_loader = DataLoader(
            test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False, num_workers=0,
        )
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    if isinstance(layer_names, tuple):
        # per-layer channel_dim (e.g. vit_full_target_layers) overrides cfg["channel_dim"]
        layer_names, channel_dim = layer_names

    ranks = _get_ranks(
        name, variant, ranking, cfg, model, train_set, layer_names, channel_dim, clip_percentile, device, calib_seed,
        ctx, channel_subsample=channel_subsample,
    )

    tag = f"variant={variant}, ranking={ranking}, calib_seed={calib_seed}"
    print(f"\n[{tag}] Table 2 (accuracy by method x bits):")
    with ctx.timed("table2"):
        table2 = compare_methods(
            model, test_loader, layer_names, ranks,
            bit_levels=(3, 4, 5), m=3, r=cfg["r"], device=device, repeats=1, max_batches=max_batches,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
        )
    for method, by_bits in table2.items():
        row = "  ".join(f"{b}b={acc*100:5.2f}%" for b, (acc, _) in by_bits.items())
        print(f"  {method:12s} {row}")
    ctx.log_table2(table2)

    print(f"\n[{tag}] Figure 3 (accuracy vs r, 3-bit):")
    with ctx.timed("figure3"):
        fig3 = sweep_ratio(
            model, test_loader, layer_names, ranks, n_bits=3, m=3, device=device, max_batches=max_batches,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
        )
    print("  " + "  ".join(f"r={r:.1f}:{a*100:5.2f}%" for r, a in fig3.items()))
    ctx.log_figure3(fig3, m=3)

    print(f"\n[{tag}] Figure 4 (accuracy vs m, 3-bit, r={cfg['r']:.2f}):")
    with ctx.timed("figure4"):
        fig4 = sweep_extra_bits(
            model, test_loader, layer_names, ranks, n_bits=3, r=cfg["r"], device=device, max_batches=max_batches,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
        )
    print("  " + "  ".join(f"m={m}:{a*100:5.2f}%" for m, a in fig4.items()))
    ctx.log_figure4(fig4, r=cfg["r"])

    ctx.flush()
    return table2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="resnet18_tinyimagenet")
    ap.add_argument("--variant", choices=list(VARIANTS), default="clip_p999")
    ap.add_argument("--ranking", choices=list(RANKINGS), default="greedy")
    ap.add_argument("--seed", type=int, default=0, help="global RNG seed, for NoisyQuant's noise search")
    ap.add_argument("--calib-seed", type=int, default=0, help="which calibration subset to draw for ranking")
    ap.add_argument("--max-batches", type=int, default=8, help="test batches per accuracy eval; 0 = full test set")
    ap.add_argument(
        "--channel-subsample", type=int, default=0,
        help="greedy-only: candidate channels evaluated per layer (0 = default = full Algorithm 3, no "
             "subsampling). Pass a positive int (e.g. 32) to opt into the speed shortcut -- it is degenerate "
             "whenever round(C*r) exceeds that value for a layer (see _get_ranks docstring), so only use it "
             "knowingly, after checking the model's channel counts.",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    run(
        args.experiment, args.variant, args.ranking,
        max_batches=(None if args.max_batches == 0 else args.max_batches),
        seed=args.seed, calib_seed=args.calib_seed,
        channel_subsample=(None if args.channel_subsample == 0 else args.channel_subsample),
    )


if __name__ == "__main__":
    main()
