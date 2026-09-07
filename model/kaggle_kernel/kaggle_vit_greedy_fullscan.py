"""
kaggle_vit_greedy_fullscan.py
=================================================

Full-scan (channel_subsample=None) greedy Algorithm 3 ranking + Table2/Figure3/
Figure4 for ViT-b-16/TinyImageNet, meant to run on Kaggle (local estimate: single
GPU, ~3.15s/candidate x 12 layers x 768 channels = ~8h ranking alone -- see the
2026-07-19 discussion in this project's chat history; Kaggle's faster GPUs and/or
T4x2 should cut this substantially, hence running it there instead of locally).

Unlike model/kaggle_finetune_vit.py, this one reuses this project's actual `quant`
package (Algorithm 1/2/3, hooks, evaluate.*) instead of reimplementing anything --
much less error-prone than hand-porting the quantization math. Before running,
attach three Kaggle Datasets:
  1. edqa-quant-package        -- this project's quant/ package. Uploaded as a zip
                                   (kaggle datasets create --dir-mode zip); Kaggle
                                   auto-extracts zips on ingestion but has been
                                   observed to flatten the quant/ prefix away (.py
                                   files land directly in the dataset root, whose
                                   own name has a hyphen and isn't an importable
                                   package name anyway) -- the script below copies
                                   whatever it finds into a properly-named quant/
                                   folder under /kaggle/working/ at runtime, so this
                                   works regardless of which layout Kaggle gives you.
  2. vit-b16-tinyimagenet-ckpt -- the 84.63%-val_acc checkpoint (updated 2026-07-19
                                   after the Kaggle continued-training rounds)
  3. tiny-imagenet-200-imagefolder -- this project's data/tiny-imagenet-200/,
                                   already in ImageFolder layout

Check the actual mounted paths in the notebook's Data panel (or run the default
os.walk('/kaggle/input') cell) and fix QUANT_ZIP / CKPT_PATH / DATA_ROOT below if
they don't match -- Kaggle's exact /kaggle/input/... layout has moved around before
in this project (see the CKPT_PATH correction in model/kaggle_finetune_vit.py's
history).

Use "Save Version" -> "Save & Run All (Commit)" for background execution (same
reasoning as model/kaggle_finetune_vit.py). Results print to the log; the full
rank table also gets written to /kaggle/working/ranks_vit_b16_tinyimagenet_
greedy_fullscan.json if you want to download and diff it against the energy-
ranking result already computed locally.
"""

import sys
import os
import shutil
import time

# ---- adjust these three to match your attached datasets' mounted paths ----
# QUANT_DATASET_DIR is wherever the quant/*.py files actually landed -- Kaggle's zip
# handling has been inconsistent in this project (sometimes preserves the "quant/"
# folder prefix, sometimes flattens the .py files straight into the dataset root, whose
# name has a hyphen and so isn't a legal Python package name either way). Rather than
# depend on exactly which layout shows up, always rebuild a properly-named quant/
# package under /kaggle/working/ at runtime, so the existing relative imports
# (`from .hooks import ...` etc inside the quant/*.py files) still work regardless.
QUANT_DATASET_DIR = "/kaggle/input/datasets/nexuslrak/edqa-quant-package"
CKPT_PATH = "/kaggle/input/datasets/nexuslrak/vit-b16-tinyimagenet-ckpt/vit_b16_tinyimagenet.pt"
DATA_ROOT = "/kaggle/input/datasets/nexuslrak/tiny-imagenet-200-imagefolder"

QUANT_PKG_ROOT = "/kaggle/working/quant_pkg"
_dest = os.path.join(QUANT_PKG_ROOT, "quant")
os.makedirs(_dest, exist_ok=True)
# handle both possible layouts: .py files directly in QUANT_DATASET_DIR, or nested
# one level down under QUANT_DATASET_DIR/quant/
_src = os.path.join(QUANT_DATASET_DIR, "quant")
if not os.path.isdir(_src):
    _src = QUANT_DATASET_DIR
for _fname in os.listdir(_src):
    if _fname.endswith(".py"):
        shutil.copy(os.path.join(_src, _fname), os.path.join(_dest, _fname))
print(f"staged {len(os.listdir(_dest))} .py files into {_dest} (source: {_src})")

sys.path.insert(0, QUANT_PKG_ROOT)

import torch
import torch.nn as nn

from quant.hooks import vit_encoder_only_target_layers
from quant.ranking import rank_channels, save_ranks
from quant.data import tinyimagenet_loaders, TINYIMAGENET_CALIB_SIZE, calibration_loader
from quant.evaluate import compare_methods, sweep_ratio, sweep_extra_bits

R = 0.40
M = 3
N_BITS = 3
CLIP_PERCENTILE = 99.9
OUT_RANKS_PATH = "/kaggle/working/ranks_vit_b16_tinyimagenet_greedy_fullscan.json"

# 2026-07-19: vit_mlp_target_layers (12 of 37 activation-producing layers, only
# each block's final MLP output) was found to badly under-quantize ViT -- Direct
# 3-bit barely degraded accuracy at all (82.26% vs 84.63% full precision) vs the
# paper's 73pp collapse, because it skips every self-attention output AND the
# 3072-dim pre-GELU intermediate MLP activation entirely.
#
# 2026-07-21: the paper's authors confirmed by email that they only quantized
# activations inside the EncoderBlocks (not conv_proj / heads.head), and that
# "channels" means the 3rd dim (0-indexed dim=2) of the activation tensor --
# i.e. exactly channel_dim=2 on the (B, N_tokens, C) EncoderBlock activations.
# This script now uses vit_encoder_only_target_layers (36 layers: self_attention
# + mlp.0 + mlp per block) to match that scope exactly, as a separate pipeline
# from the project's main energy-ranking results, which stay on the broader
# 38-layer vit_full_target_layers scope and are not being re-run.


def build_model():
    from torchvision.models import vit_b_16
    model = vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 200)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, gpus={torch.cuda.device_count()}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"loaded checkpoint from {CKPT_PATH}")

    train_loader, test_loader, train_set = tinyimagenet_loaders(DATA_ROOT)
    layer_names, channel_dim = vit_encoder_only_target_layers(model)
    print(f"{len(layer_names)} target layers, channel_dim={channel_dim}")

    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=0)

    print("\nranking (full-scan greedy, channel_subsample=None)...")
    t0 = time.time()
    ranks = rank_channels(
        model, calib, layer_names, n_bits=N_BITS, device=device, channel_dim=channel_dim,
        channel_subsample=None, max_batches=4,
        progress=lambda l, d, t: print(f"  ranking {l}: {d}/{t}", end="\r") if d % 32 == 0 else None,
    )
    print(f"\nranking took {time.time() - t0:.1f}s")
    save_ranks(ranks, OUT_RANKS_PATH)
    print(f"ranks saved to {OUT_RANKS_PATH}")

    print("\nTable 2 (accuracy by method x bits):")
    t0 = time.time()
    table2 = compare_methods(
        model, test_loader, layer_names, ranks,
        bit_levels=(3, 4, 5), m=M, r=R, device=device, repeats=1, max_batches=None,
        channel_dim=channel_dim, clip_percentile=CLIP_PERCENTILE,
    )
    for method, by_bits in table2.items():
        row = "  ".join(f"{b}b={acc * 100:5.2f}%" for b, (acc, _) in by_bits.items())
        print(f"  {method:12s} {row}")
    print(f"  [timing] table2: {time.time() - t0:.1f}s")

    print("\nFigure 3 (accuracy vs r, 3-bit):")
    t0 = time.time()
    fig3 = sweep_ratio(
        model, test_loader, layer_names, ranks, n_bits=N_BITS, m=M, device=device, max_batches=None,
        channel_dim=channel_dim, clip_percentile=CLIP_PERCENTILE,
    )
    print("  " + "  ".join(f"r={r:.1f}:{a * 100:5.2f}%" for r, a in fig3.items()))
    print(f"  [timing] figure3: {time.time() - t0:.1f}s")

    print(f"\nFigure 4 (accuracy vs m, 3-bit, r={R:.2f}):")
    t0 = time.time()
    fig4 = sweep_extra_bits(
        model, test_loader, layer_names, ranks, n_bits=N_BITS, r=R, device=device, max_batches=None,
        channel_dim=channel_dim, clip_percentile=CLIP_PERCENTILE,
    )
    print("  " + "  ".join(f"m={m}:{a * 100:5.2f}%" for m, a in fig4.items()))
    print(f"  [timing] figure4: {time.time() - t0:.1f}s")

    print("\ndone.")


if __name__ == "__main__":
    main()
