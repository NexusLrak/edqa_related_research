"""

Shows how the modules fit together for each (model, dataset) pair in the paper.
Ranking is the expensive step, so we build ranks once per experiment, cache them
to disk, and reuse.

    python -m quant.run_experiments --experiment resnet32_cifar10
    python -m quant.run_experiments --experiment resnet18_tinyimagenet
    python -m quant.run_experiments --experiment vit_b16_tinyimagenet
    python -m quant.run_experiments --experiment mobilenetv2_cifar10

Paper reference points: ResNet-32/MobileNetV2 on CIFAR-10; ResNet-18/ViT-b-16 on
TinyImageNet. m = 3; r = 0.55 for ResNet-18, 0.40 otherwise; 3/4/5-bit levels.

ResNet-32 and MobileNetV2 on CIFAR-10 use third-party pretrained checkpoints (no
local training needed, see their build_* docstrings). ResNet-18 and ViT-b-16 were
fine-tuned locally via model/finetune_tinyimagenet.py; see their build_* docstrings
below for the val accuracy reached.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import (
    CIFAR10_CALIB_SIZE,
    TINYIMAGENET_CALIB_SIZE,
    calibration_loader,
    cifar10_loaders,
    tinyimagenet_loaders,
)
from .evaluate import compare_methods, sweep_extra_bits, sweep_ratio
from .hooks import default_target_layers, vit_full_target_layers
from .ranking import load_ranks, rank_channels, save_ranks
from .results_logger import RunContext

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(_PROJ_ROOT, "model", "checkpoints")
TINYIMAGENET_ROOT = os.path.join(_PROJ_ROOT, "data", "tiny-imagenet-200")


def build_resnet32_cifar10():
    """ResNet-32 / CIFAR-10 via a third-party pretrained checkpoint (no local training needed)."""
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet32", pretrained=True)
    return model.eval()


def build_mobilenetv2_cifar10():
    """MobileNetV2 / CIFAR-10, same third-party repo as resnet32 (no local training needed).
    Full test set: 93.61% (verified 2026-07-19).

    Much wider than resnet32 (up to 1280 channels vs 64) -- at r=0.40, 40/53 target
    layers need more than 32 "important" channels, so ranking.rank_channels' default
    channel_subsample=32 would be degenerate here (see the 2026-07-18 ResNet-18/
    TinyImageNet finding). Use channel_subsample=None (or a larger value covering the
    widest layer's k) for a non-degenerate greedy ranking on this model.
    """
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_mobilenetv2_x1_0", pretrained=True)
    return model.eval()


def build_resnet18_tinyimagenet():
    """ResNet-18 / TinyImageNet-200, fine-tuned from ImageNet weights.

    See model/finetune_tinyimagenet.py — 10-epoch fine-tune reached 72.52% val
    accuracy (paper's "Ori Acc" for this pair is 72.51%).
    """
    from torchvision.models import resnet18
    model = resnet18(num_classes=200)
    ckpt = os.path.join(CHECKPOINT_DIR, "resnet18_tinyimagenet.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def build_vit_b16_tinyimagenet():
    """ViT-b-16 / TinyImageNet-200, fine-tuned from ImageNet weights.

    10-epoch local fine-tune reached 83.86% val accuracy; continued training on
    Kaggle (T4x2, model/kaggle_finetune_vit.py, low peak LR 2e-5/1e-5/5e-6 across
    several --resume rounds to avoid the cosine-restart regression seen at the
    original 1e-4 -- see quant/README.md), finishing with 20 epochs at a low
    LR (3e-6); the last 10 of those epochs plateaued around 84.5% (final:
    84.63-84.65%, verified 2026-07-19). This is a genuine plateau, not an
    under-trained checkpoint -- still well short of the paper's "Ori Acc" of
    88.31%, plausibly because the fine-tuning recipe itself (AdamW, cosine,
    these epoch counts/LRs) isn't what the paper used -- their exact ViT
    hyperparameters aren't published. Treat the 88.31% gap as a known reproduction
    limitation, not a bug to keep chasing indefinitely.

    Separately (2026-07-19): all ViT numbers produced before this date used
    vit_mlp_target_layers, which only hooks each block's final MLP output (12 of
    37 activation-producing layers) -- see that function's docstring for why this
    was a real bug (near-zero Direct-quantization degradation vs the paper's
    73pp collapse) and not just noise. EXPERIMENTS now uses vit_full_target_layers
    instead; any ViT numbers recorded before this fix should be treated as stale.
    """
    from torchvision.models import vit_b_16
    model = vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 200)
    ckpt = os.path.join(CHECKPOINT_DIR, "vit_b16_tinyimagenet.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


# name -> config: model builder, layer-names fn, channel dim, loaders fn, calib size, r
EXPERIMENTS = {
    "resnet32_cifar10": dict(
        build=build_resnet32_cifar10,
        target_layers=default_target_layers,
        channel_dim=1,
        loaders=lambda: cifar10_loaders(root="./data"),
        calib_size=CIFAR10_CALIB_SIZE,
        r=0.40,
    ),
    "resnet18_tinyimagenet": dict(
        build=build_resnet18_tinyimagenet,
        target_layers=default_target_layers,
        channel_dim=1,
        loaders=lambda: tinyimagenet_loaders(TINYIMAGENET_ROOT),
        calib_size=TINYIMAGENET_CALIB_SIZE,
        r=0.55,
    ),
    "vit_b16_tinyimagenet": dict(
        build=build_vit_b16_tinyimagenet,
        target_layers=vit_full_target_layers,  # returns (names, channel_dims) — see hooks.py
        channel_dim=2,                         # placeholder; overridden by target_layers' own dict, see run()
        loaders=lambda: tinyimagenet_loaders(TINYIMAGENET_ROOT),
        calib_size=TINYIMAGENET_CALIB_SIZE,
        r=0.40,
    ),
    "mobilenetv2_cifar10": dict(
        build=build_mobilenetv2_cifar10,
        target_layers=default_target_layers,
        channel_dim=1,
        loaders=lambda: cifar10_loaders(root="./data"),
        calib_size=CIFAR10_CALIB_SIZE,
        r=0.40,
    ),
}


def run(name: str, max_batches: "int | None" = 8, seed: int = 0):
    """max_batches=None evaluates the FULL test set. This matters a lot -- an
    8-batch subset can give a meaningfully different (even higher) number than
    the full set, see quant/experiment_logs/ for the 2026-07 finding where an
    8-batch eDQA read 69.34% but the full 79-batch TinyImageNet val set read
    63.20% for the identical config. Always check `eval_batches` in results.csv
    before comparing numbers across runs.
    """
    cfg = EXPERIMENTS[name]
    ctx = RunContext(experiment=name, pipeline="paper", ranking="greedy", seed=seed, eval_batches=max_batches)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = cfg["build"]().to(device)

    train_loader, test_loader, train_set = cfg["loaders"]()
    # See run_experiments_tuned.run()'s identical (and more detailed) rebuild for why
    # this is conditional on max_batches: small-subset repeated fresh iteration favors
    # num_workers=0 (avoids Windows' per-iteration respawn cost), but full-test-set runs
    # favor keeping the loaders() factory's parallel workers instead.
    if max_batches is not None:
        test_loader = DataLoader(
            test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False, num_workers=0,
        )
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    if isinstance(layer_names, tuple):
        # per-layer channel_dim (e.g. vit_full_target_layers) overrides cfg["channel_dim"]
        layer_names, channel_dim = layer_names

    rank_path = f"ranks_{name}_3bit.json"
    if os.path.exists(rank_path):
        ranks = load_ranks(rank_path)
    else:
        calib = calibration_loader(train_set, cfg["calib_size"], seed=0)
        with ctx.timed("ranking"):
            ranks = rank_channels(
                model,
                calib,
                layer_names,
                n_bits=3,
                device=device,
                channel_dim=channel_dim,
                channel_subsample=32,      # evaluate 32 candidate channels/layer for speed
                max_batches=4,             # cap calibration batches per inference
                progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", end="\r"),
            )
        save_ranks(ranks, rank_path)

    print("\nTable 2 (accuracy by method x bits):")
    with ctx.timed("table2"):
        table2 = compare_methods(
            model, test_loader, layer_names, ranks,
            bit_levels=(3, 4, 5), m=3, r=cfg["r"], device=device, repeats=1, max_batches=max_batches,
            channel_dim=channel_dim,
        )
    for method, by_bits in table2.items():
        row = "  ".join(f"{b}b={acc*100:5.2f}%" for b, (acc, _) in by_bits.items())
        print(f"  {method:12s} {row}")
    ctx.log_table2(table2)

    print("\nFigure 3 (accuracy vs r, 3-bit):")
    with ctx.timed("figure3"):
        fig3 = sweep_ratio(
            model, test_loader, layer_names, ranks, n_bits=3, m=3, device=device, max_batches=max_batches,
            channel_dim=channel_dim,
        )
    print("  " + "  ".join(f"r={r:.1f}:{a*100:5.2f}%" for r, a in fig3.items()))
    ctx.log_figure3(fig3, m=3)

    print(f"\nFigure 4 (accuracy vs m, 3-bit, r={cfg['r']:.2f}):")
    with ctx.timed("figure4"):
        fig4 = sweep_extra_bits(
            model, test_loader, layer_names, ranks, n_bits=3, r=cfg["r"], device=device, max_batches=max_batches,
            channel_dim=channel_dim,
        )
    print("  " + "  ".join(f"m={m}:{a*100:5.2f}%" for m, a in fig4.items()))
    ctx.log_figure4(fig4, r=cfg["r"])

    ctx.flush()
    return table2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=list(EXPERIMENTS), default="resnet32_cifar10")
    ap.add_argument("--seed", type=int, default=0, help="global RNG seed, for NoisyQuant's noise search")
    ap.add_argument("--max-batches", type=int, default=8, help="test batches per accuracy eval; 0 = full test set")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    run(args.experiment, max_batches=(None if args.max_batches == 0 else args.max_batches), seed=args.seed)


if __name__ == "__main__":
    main()
