"""
run.py — portable entry point for edqa_portable_package.zip
=================================================

Platform-agnostic version of quant/run_experiments_tuned.py: same ranking
methods (greedy / energy / surrogate / stratified) and same Table2/Figure3/
Figure4 pipeline, but every path is resolved relative to THIS file instead of
the original project layout, and resnet32_cifar10 / mobilenetv2_cifar10 no
longer call torch.hub.load(...) (which needs internet access to GitHub) --
they build the architecture from the vendored vendor/pytorch_cifar_models/
package and load a local checkpoint instead. Meant to be unzipped and run on
any rented GPU box (Kaggle, AutoDL, RunPod, Vast.ai, Lambda, Colab, a bare
Linux VM, ...), not just Kaggle.

Layout this script expects (all produced by packaging/build_package.py):
    run.py                  <- this file
    quant/                  <- this project's quant package, unmodified
    vendor/pytorch_cifar_models/   <- resnet32/mobilenetv2 architecture defs
    checkpoints/*.pt        <- one file per experiment, see EXPERIMENTS below
    data/                   <- cifar-10-python.tar.gz and/or tiny-imagenet-200.zip
    output/                 <- created on first run; ranks + results json land here

Usage:
    python run.py --experiment vit_b16_tinyimagenet --ranking greedy --channel-subsample 0 --max-batches 0
    python run.py --experiment resnet18_tinyimagenet --ranking energy --max-batches 0
    python run.py --experiment mobilenetv2_cifar10 --ranking greedy --channel-subsample 0 --max-batches 0

--max-batches 0 evaluates the FULL test set. The default (8) is fast-iteration
only -- an 8-batch subset can read meaningfully higher than the full set (see
quant/RESULTS.md), don't cite numbers produced with the default.

--channel-subsample 0 runs the literal, unabridged Algorithm 3 (every channel
evaluated per layer) -- this is the "greedy fullscan" mode that's normally too
slow to run locally and is the whole reason this package exists. The default
(32) is a speed shortcut; check the target model's channel counts before
trusting it (see quant/run_experiments_tuned.py's _get_ranks docstring).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PKG_ROOT, "checkpoints")
DATA_DIR = os.path.join(PKG_ROOT, "data")
OUT_DIR = os.path.join(PKG_ROOT, "output")
VENDOR_DIR = os.path.join(PKG_ROOT, "vendor")

sys.path.insert(0, PKG_ROOT)
sys.path.insert(0, VENDOR_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quant.data import (
    CIFAR10_CALIB_SIZE,
    TINYIMAGENET_CALIB_SIZE,
    calibration_loader,
    cifar10_loaders,
    tinyimagenet_loaders,
)
from quant.evaluate import compare_methods, sweep_extra_bits, sweep_ratio
from quant.hooks import (
    default_target_layers,
    vit_encoder_only_target_layers,
    vit_full_target_layers,
)
from quant.ranking import channel_count, load_ranks, rank_channels, save_ranks
from quant.rank_surrogates import (
    rank_channels_via_energy,
    rank_channels_via_stratified,
    rank_channels_via_surrogate_per_layer,
)
from quant.rank_gradient_surrogates import (
    protection_gain_scores,
    rank_channels_via_edqa_saliency,
    rank_channels_via_fisher,
    rank_channels_via_fisher_a2,
    rank_channels_via_pg2,
    rank_channels_via_saliency,
)


# --------------------------------------------------------------------------- #
# data staging -- extract/restructure on first run, no-op afterwards
# --------------------------------------------------------------------------- #

def _ensure_tinyimagenet() -> str:
    root = os.path.join(DATA_DIR, "tiny-imagenet-200")
    val_images_dir = os.path.join(root, "val", "images")
    if os.path.isdir(root) and not os.path.isdir(val_images_dir):
        return root  # already extracted and restructured

    zip_path = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
    if not os.path.isdir(root):
        if not os.path.exists(zip_path):
            raise FileNotFoundError(
                f"neither {root} nor {zip_path} exist -- see README.md's data section"
            )
        print(f"extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)

    if os.path.isdir(val_images_dir):
        print("restructuring val/ into ImageFolder layout (val/<wnid>/*.JPEG) ...")
        val_dir = os.path.join(root, "val")
        ann_file = os.path.join(val_dir, "val_annotations.txt")
        with open(ann_file) as f:
            mapping = {}
            for line in f:
                parts = line.strip().split("\t")
                mapping[parts[0]] = parts[1]
        moved = 0
        for fname, wnid in mapping.items():
            src = os.path.join(val_images_dir, fname)
            if not os.path.exists(src):
                continue
            dst_dir = os.path.join(val_dir, wnid)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, fname))
            moved += 1
        print(f"  moved {moved} images into {len(set(mapping.values()))} class folders")
        if not os.listdir(val_images_dir):
            os.rmdir(val_images_dir)
    return root


# --------------------------------------------------------------------------- #
# model builders -- no torch.hub / network access, everything from local files
# --------------------------------------------------------------------------- #

def build_resnet32_cifar10():
    from pytorch_cifar_models.resnet import cifar10_resnet32
    model = cifar10_resnet32(pretrained=False)
    ckpt = os.path.join(CHECKPOINT_DIR, "cifar10_resnet32.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def build_mobilenetv2_cifar10():
    from pytorch_cifar_models.mobilenetv2 import cifar10_mobilenetv2_x1_0
    model = cifar10_mobilenetv2_x1_0(pretrained=False)
    ckpt = os.path.join(CHECKPOINT_DIR, "cifar10_mobilenetv2_x1_0.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def build_resnet18_tinyimagenet():
    from torchvision.models import resnet18
    model = resnet18(num_classes=200)
    ckpt = os.path.join(CHECKPOINT_DIR, "resnet18_tinyimagenet.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def build_vit_b16_tinyimagenet():
    from torchvision.models import vit_b_16
    model = vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 200)
    ckpt = os.path.join(CHECKPOINT_DIR, "vit_b16_tinyimagenet.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


EXPERIMENTS = {
    "resnet32_cifar10": dict(
        build=build_resnet32_cifar10, target_layers=default_target_layers, channel_dim=1,
        loaders=lambda: cifar10_loaders(root=DATA_DIR), calib_size=CIFAR10_CALIB_SIZE, r=0.40,
    ),
    "mobilenetv2_cifar10": dict(
        build=build_mobilenetv2_cifar10, target_layers=default_target_layers, channel_dim=1,
        loaders=lambda: cifar10_loaders(root=DATA_DIR), calib_size=CIFAR10_CALIB_SIZE, r=0.40,
    ),
    "resnet18_tinyimagenet": dict(
        build=build_resnet18_tinyimagenet, target_layers=default_target_layers, channel_dim=1,
        loaders=lambda: tinyimagenet_loaders(_ensure_tinyimagenet()), calib_size=TINYIMAGENET_CALIB_SIZE, r=0.55,
    ),
    "vit_b16_tinyimagenet": dict(
        build=build_vit_b16_tinyimagenet, target_layers=vit_full_target_layers, channel_dim=2,
        loaders=lambda: tinyimagenet_loaders(_ensure_tinyimagenet()), calib_size=TINYIMAGENET_CALIB_SIZE, r=0.40,
    ),
}

VARIANTS = {
    "baseline": dict(clip_percentile=None),
    "clip_p999": dict(clip_percentile=99.9),
    "clip_p99": dict(clip_percentile=99.0),
}

RANKINGS = (
    "greedy", "surrogate", "energy", "stratified",
    "saliency", "fisher", "fisher_a2", "protection_gain", "pg2", "edqa_saliency",
)


# --------------------------------------------------------------------------- #
# ranking (cached to OUT_DIR, same cache-key conventions as run_experiments_tuned.py)
# --------------------------------------------------------------------------- #

def _rank_cache_suffix(calib_seed: int) -> str:
    return "" if calib_seed == 0 else f"_calibseed{calib_seed}"


def _get_ranks(
    name, variant, ranking, cfg, model, train_set, layer_names, channel_dim, device, calib_seed, ctx,
    channel_subsample=32,
):
    suffix = _rank_cache_suffix(calib_seed)

    if ranking == "greedy":
        subsample_suffix = "" if channel_subsample == 32 else (
            "_fullscan" if channel_subsample is None else f"_subsample{channel_subsample}"
        )
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_3bit{suffix}{subsample_suffix}.json")
        if os.path.exists(rank_path):
            existing = load_ranks(rank_path)
            if all(l in existing for l in layer_names):
                print(f"loaded complete ranks from {rank_path}, skipping ranking")
                return existing
            print(f"found partial ranks at {rank_path} ({len(existing)}/{len(layer_names)} layers) -- resuming")
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        t0 = time.time()
        ranks = rank_channels(
            model, calib, layer_names, n_bits=3, device=device, channel_dim=channel_dim,
            channel_subsample=channel_subsample, max_batches=4,
            progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", end="\r") if d % 32 == 0 else None,
            checkpoint_path=rank_path,
        )
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "surrogate":
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_surrogate_{variant}{suffix}.json")
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        _, test_loader_for_ranking, _ = cfg["loaders"]()
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        t0 = time.time()
        ranks = rank_channels_via_surrogate_per_layer(
            model, calib, test_loader_for_ranking, layer_names, device,
            channel_dim=channel_dim, clip_percentile=VARIANTS[variant]["clip_percentile"],
            r=cfg["r"], verbose=True,
        )
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "energy":
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_pure_energy_calibseed{calib_seed}.json")
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        t0 = time.time()
        ranks = rank_channels_via_energy(model, calib, layer_names, device, channel_dim=channel_dim)
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    if ranking == "stratified":
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_stratified_calibseed{calib_seed}.json")
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        t0 = time.time()
        ranks = rank_channels_via_stratified(
            model, calib, layer_names, device, r=cfg["r"], channel_dim=channel_dim,
        )
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    if ranking in ("saliency", "fisher", "fisher_a2"):
        # No clip_percentile dependence -- raw activation/gradient statistics,
        # same reasoning as "energy" above.
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_{ranking}_calibseed{calib_seed}.json")
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        fn = {"saliency": rank_channels_via_saliency, "fisher": rank_channels_via_fisher,
              "fisher_a2": rank_channels_via_fisher_a2}[ranking]
        t0 = time.time()
        ranks = fn(model, calib, layer_names, device, channel_dim=channel_dim, calib_batches=1)
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    if ranking in ("protection_gain", "pg2", "edqa_saliency"):
        # Keyed on (variant, calib_seed) like "surrogate": pg depends on
        # clip_percentile via fake_quantize_direct. Reference configuration
        # throughout this project (n=3, m=3).
        rank_path = os.path.join(OUT_DIR, f"ranks_{name}_{ranking}_{variant}{suffix}.json")
        if os.path.exists(rank_path):
            return load_ranks(rank_path)
        calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
        clip_percentile = VARIANTS[variant]["clip_percentile"]
        t0 = time.time()
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
        ctx["ranking_seconds"] = time.time() - t0
        save_ranks(ranks, rank_path)
        return ranks

    raise ValueError(f"unknown ranking {ranking!r}, choose from {RANKINGS}")


def _total_candidates(model, layer_names, channel_dim, channel_subsample) -> int:
    """Mirrors ranking.rank_channels' per-layer candidate-count logic (see its
    channel_subsample handling) without actually running anything -- used to
    extrapolate a --layers-limit probe's measured seconds/candidate up to the
    full layer set.
    """
    total = 0
    for layer in layer_names:
        dim = channel_dim[layer] if isinstance(channel_dim, dict) else channel_dim
        C = channel_count(model, layer, dim)
        total += C if not channel_subsample or channel_subsample >= C else channel_subsample
    return total


def probe_greedy_timing(
    name: str, calib_seed: int = 0, channel_subsample: "int | None" = None, layers_limit: int = 1,
    vit_scope: "str | None" = None,
):
    """--layers-limit validation run: full-scan greedy on just the FIRST
    `layers_limit` layers (in forward order), time it, then extrapolate to
    what the complete layer set would cost on this GPU. Does not touch the
    rank cache and does not run Table2/Figure3/Figure4 -- a partial rank dict
    would leave every unranked layer at full precision in the eDQA path,
    which isn't a meaningful accuracy number, only a timing measurement.

    This is the same "measure a slice, extrapolate the total" method used to
    produce the ~120h estimate documented in quant/RESULTS.md (there: one
    partially-finished Kaggle job's layer0 timing, extrapolated to 36 layers)
    -- run this BEFORE committing a rented GPU to a multi-hour/multi-day job.
    """
    cfg = EXPERIMENTS[name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, gpus={torch.cuda.device_count() if device == 'cuda' else 0}")
    model = cfg["build"]().to(device)

    _, _, train_set = cfg["loaders"]()

    target_layers_fn = cfg["target_layers"]
    if name == "vit_b16_tinyimagenet":
        scope = vit_scope or "encoder_only"
        target_layers_fn = vit_encoder_only_target_layers if scope == "encoder_only" else vit_full_target_layers

    full_layer_names = target_layers_fn(model)
    channel_dim = cfg["channel_dim"]
    if isinstance(full_layer_names, tuple):
        full_layer_names, channel_dim = full_layer_names

    layers_limit = min(layers_limit, len(full_layer_names))
    probe_layers = full_layer_names[:layers_limit]
    print(
        f"probing {layers_limit}/{len(full_layer_names)} layers "
        f"(channel_subsample={channel_subsample or 'None (fullscan)'}): {probe_layers}"
    )

    calib = calibration_loader(train_set, cfg["calib_size"], seed=calib_seed)
    t0 = time.time()
    rank_channels(
        model, calib, probe_layers, n_bits=3, device=device, channel_dim=channel_dim,
        channel_subsample=channel_subsample, max_batches=4,
        progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", end="\r") if d % 16 == 0 else None,
    )
    elapsed = time.time() - t0

    probe_candidates = _total_candidates(model, probe_layers, channel_dim, channel_subsample)
    full_candidates = _total_candidates(model, full_layer_names, channel_dim, channel_subsample)
    seconds_per_candidate = elapsed / probe_candidates
    estimated_full_seconds = seconds_per_candidate * full_candidates

    print(f"\nprobe: {probe_candidates} candidates in {elapsed:.1f}s "
          f"({seconds_per_candidate:.2f}s/candidate on this GPU)")
    print(f"extrapolated full scope ({len(full_layer_names)} layers, {full_candidates} candidates): "
          f"{estimated_full_seconds / 3600:.1f} hours")
    return seconds_per_candidate, estimated_full_seconds


# --------------------------------------------------------------------------- #
# main run
# --------------------------------------------------------------------------- #

def run(
    name: str, variant: str, ranking: str, max_batches: "int | None" = 8, seed: int = 0, calib_seed: int = 0,
    channel_subsample: "int | None" = 32, vit_scope: "str | None" = None, skip_figures: bool = False,
):
    cfg = EXPERIMENTS[name]
    clip_percentile = VARIANTS[variant]["clip_percentile"]
    ctx: dict = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, gpus={torch.cuda.device_count() if device == 'cuda' else 0}")
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    if max_batches is not None:
        test_loader = DataLoader(
            test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False, num_workers=0,
        )

    target_layers_fn = cfg["target_layers"]
    if name == "vit_b16_tinyimagenet":
        # the paper's authors confirmed (2026-07-21) the greedy search only ever
        # covers the 36 EncoderBlock layers, not conv_proj/heads.head -- use the
        # narrower scope for "greedy" unless the caller explicitly overrides it,
        # keep the broader 38-layer default for every other ranking method.
        scope = vit_scope or ("encoder_only" if ranking == "greedy" else "full")
        target_layers_fn = vit_encoder_only_target_layers if scope == "encoder_only" else vit_full_target_layers

    layer_names = target_layers_fn(model)
    channel_dim = cfg["channel_dim"]
    if isinstance(layer_names, tuple):
        layer_names, channel_dim = layer_names
    print(f"{len(layer_names)} target layers, channel_dim={channel_dim}")

    ranks = _get_ranks(
        name, variant, ranking, cfg, model, train_set, layer_names, channel_dim, device, calib_seed, ctx,
        channel_subsample=channel_subsample,
    )

    tag = f"experiment={name}, variant={variant}, ranking={ranking}, calib_seed={calib_seed}"
    print(f"\n[{tag}] Table 2 (accuracy by method x bits):")
    t0 = time.time()
    table2 = compare_methods(
        model, test_loader, layer_names, ranks,
        bit_levels=(3, 4, 5), m=3, r=cfg["r"], device=device, repeats=1, max_batches=max_batches,
        channel_dim=channel_dim, clip_percentile=clip_percentile,
    )
    ctx["table2_seconds"] = time.time() - t0
    for method, by_bits in table2.items():
        row = "  ".join(f"{b}b={acc * 100:5.2f}%" for b, (acc, _) in by_bits.items())
        print(f"  {method:12s} {row}")

    if skip_figures:
        print(f"\n[{tag}] Figure 3 / Figure 4: skipped (--skip-figures)")
        fig3, fig4 = {}, {}
    else:
        print(f"\n[{tag}] Figure 3 (accuracy vs r, 3-bit):")
        t0 = time.time()
        fig3 = sweep_ratio(
            model, test_loader, layer_names, ranks, n_bits=3, m=3, device=device, max_batches=max_batches,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
        )
        ctx["figure3_seconds"] = time.time() - t0
        print("  " + "  ".join(f"r={r:.1f}:{a * 100:5.2f}%" for r, a in fig3.items()))

        print(f"\n[{tag}] Figure 4 (accuracy vs m, 3-bit, r={cfg['r']:.2f}):")
        t0 = time.time()
        fig4 = sweep_extra_bits(
            model, test_loader, layer_names, ranks, n_bits=3, r=cfg["r"], device=device, max_batches=max_batches,
            channel_dim=channel_dim, clip_percentile=clip_percentile,
        )
        ctx["figure4_seconds"] = time.time() - t0
        print("  " + "  ".join(f"m={m}:{a * 100:5.2f}%" for m, a in fig4.items()))

    out_path = os.path.join(
        OUT_DIR, f"results_{name}_{ranking}_{variant}_calibseed{calib_seed}.json"
    )
    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": name, "variant": variant, "ranking": ranking,
                "seed": seed, "calib_seed": calib_seed, "max_batches": max_batches,
                "channel_subsample": channel_subsample,
                "table2": {m: {str(b): acc for b, (acc, _) in bb.items()} for m, bb in table2.items()},
                "figure3": {str(r): a for r, a in fig3.items()},
                "figure4": {str(m): a for m, a in fig4.items()},
                "timings_seconds": ctx,
            },
            f, indent=2,
        )
    print(f"\nresults written to {out_path}")
    return table2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="resnet18_tinyimagenet")
    ap.add_argument("--variant", choices=list(VARIANTS), default="clip_p999")
    ap.add_argument("--ranking", choices=list(RANKINGS), default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=8, help="0 = full test set")
    ap.add_argument(
        "--channel-subsample", type=int, default=32,
        help="greedy-only: candidates evaluated per layer, 0 = full Algorithm 3 (fullscan)",
    )
    ap.add_argument(
        "--vit-scope", choices=("full", "encoder_only"), default=None,
        help="vit_b16_tinyimagenet only: which target-layer set to use; default auto-picks "
             "encoder_only for --ranking greedy (paper-aligned) and full otherwise",
    )
    ap.add_argument(
        "--skip-figures", action="store_true",
        help="skip Figure3 (accuracy vs r) and Figure4 (accuracy vs m) sweeps -- these reproduce "
             "the original eDQA paper's own figures (fixed ranking, varying r/m) and don't add "
             "new information for a ranking-method comparison beyond what Table2's single "
             "edqa row already gives; only Table2 (accuracy by method x bits) is computed.",
    )
    ap.add_argument(
        "--layers-limit", type=int, default=None,
        help="VALIDATION MODE: only full-scan the first N layers (forward order), time it, "
             "extrapolate to the complete layer set, then exit -- does NOT run Table2/Figure3/"
             "Figure4 (a partial rank dict isn't a meaningful accuracy number, only a timing "
             "one). Run this on a new/untested GPU box BEFORE committing to the real "
             "--channel-subsample 0 run, e.g.: "
             "--experiment vit_b16_tinyimagenet --ranking greedy --channel-subsample 0 --layers-limit 1",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    if args.layers_limit is not None:
        probe_greedy_timing(
            args.experiment, calib_seed=args.calib_seed,
            channel_subsample=(None if args.channel_subsample == 0 else args.channel_subsample),
            layers_limit=args.layers_limit, vit_scope=args.vit_scope,
        )
        return

    run(
        args.experiment, args.variant, args.ranking,
        max_batches=(None if args.max_batches == 0 else args.max_batches),
        seed=args.seed, calib_seed=args.calib_seed,
        channel_subsample=(None if args.channel_subsample == 0 else args.channel_subsample),
        vit_scope=args.vit_scope, skip_figures=args.skip_figures,
    )


if __name__ == "__main__":
    main()
