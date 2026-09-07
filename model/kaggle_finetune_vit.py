"""
kaggle_finetune_vit.py
=================================================

Self-contained (no dependency on the `quant` package) version of
finetune_tinyimagenet.py's ViT-b-16 continuation, meant to be pasted directly
into a Kaggle Notebook cell. Kaggle gives a P100 or 2xT4 (16GB+ VRAM each,
vs the 8GB laptop GPU this was originally run on) with a free weekly GPU-hour
quota -- see the 2026-07-19 discussion in this project's chat history for why.

Before running, attach two Kaggle Datasets to the notebook:
  1. tiny-imagenet-200      -- this project's data/tiny-imagenet-200/ folder
                                (already restructured to ImageFolder layout:
                                train/<wnid>/images/*.JPEG, val/<wnid>/*.JPEG)
  2. vit-b16-tinyimagenet-ckpt -- this project's model/checkpoints/vit_b16_tinyimagenet.pt
                                (the 83.86%-val_acc checkpoint to continue from)

Then fix DATA_ROOT / CKPT_PATH below to match the actual mounted paths (check
the right-hand "Data" panel in the notebook for the exact folder names Kaggle
assigned -- usually /kaggle/input/<dataset-slug>/...).

IMPORTANT: to actually get background execution (keep training after you
close the browser tab), use "Save Version" -> "Save & Run All (Commit)", not
just running cells interactively -- Kaggle runs committed versions as a
detached batch job. The resulting checkpoint will be in the notebook's
Output under /kaggle/working/vit_b16_tinyimagenet.pt; download it from there
and drop it into this project's model/checkpoints/ to resume the quant
pipeline / further local fine-tuning.

Root cause fixed here (matches the 2026-07-18/19 local run's finding): the
non-Kaggle script's --resume didn't preserve optimizer/scheduler state, so a
fresh cosine-annealing restart at the full peak LR briefly wrecked an
already-converged checkpoint (83.86% -> 78.6% for several epochs before slow
recovery). This version uses a LOW peak LR (2e-5, not 1e-4) for exactly that
reason -- see the best_acc-floor safety check below too, which refuses to
overwrite the checkpoint with anything worse than what you started with.
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- adjust these two to match your attached datasets' mounted paths ----
DATA_ROOT = "/kaggle/input/datasets/nexuslrak/tiny-imagenet-200-imagefolder"
CKPT_PATH = "/kaggle/input/datasets/nexuslrak/vit-b16-tinyimagenet-ckpt/vit_b16_tinyimagenet.pt"
OUT_PATH = "/kaggle/working/vit_b16_tinyimagenet.pt"

NUM_CLASSES = 200
EPOCHS = 4
LR = 5e-6
BATCH_SIZE = 128
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 4
IMAGE_SIZE = 224


def tinyimagenet_transforms(train: bool):
    from torchvision import transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def tinyimagenet_loaders(root: str):
    from torchvision import datasets
    train = datasets.ImageFolder(f"{root}/train", transform=tinyimagenet_transforms(train=True))
    val = datasets.ImageFolder(f"{root}/val", transform=tinyimagenet_transforms(train=False))
    train_loader = DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    return train_loader, val_loader


def build_model():
    from torchvision.models import vit_b_16
    model = vit_b_16()  # weights loaded from CKPT_PATH below, not ImageNet defaults
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    return model


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(1, total)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_gpus = torch.cuda.device_count()
    print(f"device={device}, gpus={n_gpus}")

    train_loader, val_loader = tinyimagenet_loaders(DATA_ROOT)

    # Load the checkpoint into the PLAIN model before any DataParallel wrapping --
    # DataParallel prefixes state_dict keys with "module.", which would silently
    # fail to match this checkpoint's keys (saved from a plain model) and would
    # also make the saved-out checkpoint incompatible with everything downstream
    # that loads it into a plain model (this project's quant pipeline, etc).
    model = build_model().to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    print(f"loaded checkpoint from {CKPT_PATH}")
    best_acc = evaluate(model, val_loader, device)
    print(f"resumed checkpoint val_acc={best_acc * 100:.2f}% (floor -- won't overwrite with worse)")

    if n_gpus > 1:
        model = nn.DataParallel(model)
        print(f"wrapped in nn.DataParallel across {n_gpus} GPUs")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    for epoch in range(EPOCHS):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running_loss += loss.item() * y.numel()
            seen += y.numel()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / max(1, seen):.4f} elapsed={time.time() - t0:.1f}s", flush=True)
        sched.step()

        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running_loss / max(1, seen):.4f} "
              f"val_acc={val_acc * 100:.2f}% time={time.time() - t0:.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            # unwrap DataParallel before saving -- keep the checkpoint's keys in
            # the plain (non "module."-prefixed) format, see the load comment above
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, OUT_PATH)
            print(f"  saved new best checkpoint to {OUT_PATH} (val_acc={val_acc * 100:.2f}%)")

    print(f"done. best val_acc={best_acc * 100:.2f}%, checkpoint at {OUT_PATH}")


if __name__ == "__main__":
    main()
