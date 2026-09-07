"""
kaggle_vit_energy_saliency_blend.py
=================================================

ViT-b-16/TinyImageNet: does a weighted rank-blend of energy and saliency
(w_e*energy_rank + w_s*saliency_rank, w_s = 1-w_e) beat either pure signal?
Local results so far (seed0, r=0.40, 3-bit): pure energy=84.42%, pure
saliency=84.97%; ResNet-18 (Ori Acc 72.52%) showed a clean monotonic
"more energy = better" curve with no beneficial blend; ResNet-32/MobileNetV2
(Ori Acc 93%+) showed no *reliable* multi-seed blend benefit either (a
seed0-only apparent peak did not replicate on seeds 1,2). This run tests
whether ViT (Ori Acc 84.63%, in between) shows a different pattern.

Same reasoning as kaggle_vit_greedy_fullscan.py for why this runs on Kaggle
(local ViT full-test evals take ~17-18 min each on a 4060-class GPU; T4 is
slower, budget more). Reuses this project's quant package rather than
reimplementing anything. Before running, attach the same three Kaggle
Datasets as kaggle_vit_greedy_fullscan.py (edqa-quant-package,
vit-b16-tinyimagenet-ckpt, tiny-imagenet-200-imagefolder) and fix
QUANT_ZIP / CKPT_PATH / DATA_ROOT below if the mounted paths differ.
Checkpoint confirmed in sync with local (343.87 MB, matches
model/kaggle_upload/vit_b16_tinyimagenet.pt) as of 2026-07-24.

Use "Save Version" -> "Save & Run All (Commit)" for background execution.
6 points (w_e = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0) x ~20-35 min/eval on a T4 =>
budget roughly 2-3.5 hours total; well within Kaggle's session limit.
"""

import sys
import os
import shutil
import time

# ---- adjust these three to match your attached datasets' mounted paths ----
QUANT_DATASET_DIR = "/kaggle/input/datasets/nexuslrak/edqa-quant-package"
CKPT_PATH = "/kaggle/input/datasets/nexuslrak/vit-b16-tinyimagenet-ckpt/vit_b16_tinyimagenet.pt"
DATA_ROOT = "/kaggle/input/datasets/nexuslrak/tiny-imagenet-200-imagefolder"

QUANT_PKG_ROOT = "/kaggle/working/quant_pkg"
_dest = os.path.join(QUANT_PKG_ROOT, "quant")
os.makedirs(_dest, exist_ok=True)
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
import torch.nn.functional as F

from quant.compression import get_compressor
from quant.hooks import QuantManager, vit_full_target_layers
from quant.data import tinyimagenet_loaders, TINYIMAGENET_CALIB_SIZE, calibration_loader
from quant.ranking import evaluate_accuracy

R = 0.40
M = 3
N_BITS = 3
CLIP_PERCENTILE = 99.9
CALIB_BATCHES = 1  # matches the official energy/saliency setup used for the 84.42%/84.97% seed0 references
SEED = 0
WEIGHTS = [(0.0, 1.0), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2), (1.0, 0.0)]  # (w_energy, w_saliency)


def build_model():
    from torchvision.models import vit_b_16
    model = vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, 200)
    return model


def channel_sum(t):
    if t.dim() == 4:
        return t.sum(dim=(0, 2, 3))
    if t.dim() == 3:
        return t.sum(dim=(0, 2))
    if t.dim() == 2:
        return t.sum(dim=0)
    raise ValueError(f"unexpected ndim {t.dim()}")


def compute_energy_and_saliency_scores(model, calib_loader, layer_names, channel_dim, device):
    modules = dict(model.named_modules())
    activations = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad()
            activations[name] = t
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    energy_accum = {name: None for name in layer_names}
    saliency_accum = {name: None for name in layer_names}

    for i, (x, y) in enumerate(calib_loader):
        if i >= CALIB_BATCHES:
            break
        model.zero_grad(set_to_none=True)
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        for name in layer_names:
            act = activations[name]
            grad = act.grad
            dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
            a = act.detach().float()
            g = grad.detach().float()
            if dim != 1:
                a = a.movedim(dim, 1)
                g = g.movedim(dim, 1)
            e = channel_sum(a.pow(2)).cpu()
            s = channel_sum(a.abs() * g.abs()).cpu()
            energy_accum[name] = e if energy_accum[name] is None else energy_accum[name] + e
            saliency_accum[name] = s if saliency_accum[name] is None else saliency_accum[name] + s

    for h in handles:
        h.remove()

    # IMPORTANT (found the hard way locally): the retain_grad()-based backward
    # graph must be explicitly freed before the eval pass below, otherwise GPU
    # memory fragmentation silently degrades the following eval loop by ~6x
    # (92s/batch instead of ~13-15s/batch). gc.collect() + empty_cache() fixes it.
    import gc
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {name: (energy_accum[name], saliency_accum[name]) for name in layer_names}


def importance_rank(scores):
    return scores.argsort(descending=True).argsort().float()


def blended_ranks(scores_by_layer, w_e, w_s):
    ranks = {}
    for name, (e, s) in scores_by_layer.items():
        er = importance_rank(e)
        sr = importance_rank(s)
        combined = w_e * er + w_s * sr
        ranks[name] = combined.argsort(descending=False).tolist()
    return ranks


def eval_with_ranks(model, test_loader, layer_names, channel_dim, ranks, device):
    comp = get_compressor("identity")
    with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=CLIP_PERCENTILE) as mgr:
        for name in layer_names:
            if name in ranks:
                mgr.set_edqa(name, N_BITS, M, R, ranks[name], comp)
        return evaluate_accuracy(model, test_loader, device=device, max_batches=None)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, gpus={torch.cuda.device_count()}")

    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print(f"loaded checkpoint from {CKPT_PATH}")

    train_loader, test_loader, train_set = tinyimagenet_loaders(DATA_ROOT)
    layer_names, channel_dim = vit_full_target_layers(model)
    print(f"{len(layer_names)} target layers")

    calib = calibration_loader(train_set, TINYIMAGENET_CALIB_SIZE, seed=SEED)

    print("\ncomputing energy + saliency scores (one forward+backward pass)...")
    t0 = time.time()
    scores = compute_energy_and_saliency_scores(model, calib, layer_names, channel_dim, device)
    print(f"scores computed ({time.time() - t0:.1f}s)\n")

    print("reference (local, seed0): pure energy=84.42%  pure saliency=84.97%\n")

    results = {}
    for w_e, w_s in WEIGHTS:
        ranks = blended_ranks(scores, w_e, w_s)
        t0 = time.time()
        acc = eval_with_ranks(model, test_loader, layer_names, channel_dim, ranks, device)
        dt = time.time() - t0
        results[(w_e, w_s)] = acc
        print(f"w_e={w_e:.1f} w_s={w_s:.1f}: {acc * 100:.2f}%  ({dt:.1f}s)")

    print("\n--- summary ---")
    for (w_e, w_s), acc in results.items():
        print(f"  w_e={w_e:.1f}: {acc * 100:.2f}%")
    print("\ndone.")


if __name__ == "__main__":
    main()
