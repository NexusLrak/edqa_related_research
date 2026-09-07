"""finetune_tinyimagenet.py — fine-tune ImageNet-pretrained backbones on TinyImageNet-200.

The eDQA paper only quantizes activations; it assumes weights are already trained.
This script produces the two checkpoints run_experiments.py needs for the
TinyImageNet side of Table 2 (ResNet-18, ViT-b-16), by replacing the ImageNet
1000-way head with a 200-way head and fine-tuning end-to-end.

Usage:
    python finetune_tinyimagenet.py --arch resnet18 --epochs 15 --out resnet18_tinyimagenet.pt
    python finetune_tinyimagenet.py --arch vit_b_16 --epochs 10 --batch-size 64 --out vit_b16_tinyimagenet.pt

TinyImageNet images are natively 64x64; quant/data.py's tinyimagenet_loaders
upsamples to 224x224 (RandomResizedCrop for train, Resize+CenterCrop for eval)
and uses ImageNet normalization stats so the pretrained backbones stay in their
native input distribution.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant.data import tinyimagenet_loaders  # noqa: E402

NUM_CLASSES = 200


def build_model(arch: str) -> nn.Module:
    if arch == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model
    if arch == "vit_b_16":
        from torchvision.models import ViT_B_16_Weights, vit_b_16
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
        return model
    raise ValueError(f"unknown arch '{arch}', choose resnet18 or vit_b_16")


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["resnet18", "vit_b_16"], required=True)
    ap.add_argument("--data-root", default=os.path.join("..", "data", "tiny-imagenet-200"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None,
                     help="debug/smoke-test: cap steps per epoch")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.normpath(os.path.join(script_dir, args.data_root))

    train_loader, val_loader, _ = tinyimagenet_loaders(
        data_root, batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = build_model(args.arch).to(device)
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"resumed from {args.resume}")
        best_acc = evaluate(model, val_loader, device)
        print(f"  resumed checkpoint val_acc={best_acc * 100:.2f}% (this is the floor -- won't overwrite with worse)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running_loss, seen = 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if args.max_steps_per_epoch and step >= args.max_steps_per_epoch:
                break
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
                elapsed = time.time() - t0
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"loss={running_loss / max(1, seen):.4f} elapsed={elapsed:.1f}s", flush=True)
        sched.step()

        val_acc = evaluate(model, val_loader, device)
        epoch_time = time.time() - t0
        print(f"epoch {epoch}: train_loss={running_loss / max(1, seen):.4f} "
              f"val_acc={val_acc * 100:.2f}% time={epoch_time:.1f}s", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)
            print(f"  saved new best checkpoint to {args.out} (val_acc={val_acc * 100:.2f}%)")

    print(f"done. best val_acc={best_acc * 100:.2f}%, checkpoint at {args.out}")


if __name__ == "__main__":
    main()
