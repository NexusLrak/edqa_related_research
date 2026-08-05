"""
data.py
=================================================

CIFAR-10 and TinyImageNet loaders plus calibration-subset sampling, matching the
paper's setup:
  * batch size = 128 for all experiments
  * calibration subset for ranking: 5000 images (CIFAR-10), 2500 (TinyImageNet)
  * every randomized experiment is repeated 5x and averaged (see evaluate.py)

torchvision is used for CIFAR-10. TinyImageNet is loaded from the standard
ImageFolder layout (train/<wnid>/images/*.JPEG, val restructured into
val/<wnid>/*.JPEG) — set TINYIMAGENET_ROOT accordingly.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset

CIFAR10_CALIB_SIZE = 5000
TINYIMAGENET_CALIB_SIZE = 2500
BATCH_SIZE = 128


def _cifar_transforms():
    from torchvision import transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def _tinyimagenet_transforms(train: bool = False, image_size: int = 224):
    """TinyImageNet transforms, resized up to the ImageNet-pretrained backbones'
    native input size (ResNet-18 / ViT-b-16 both expect 224x224).

    Uses ImageNet normalization stats (not TinyImageNet-specific) since the
    backbones are ImageNet-pretrained and their BatchNorm/patch-embed statistics
    were fit to that distribution.
    """
    from torchvision import transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def cifar10_loaders(root: str = "./data", batch_size: int = BATCH_SIZE, num_workers: int = 4):
    from torchvision import datasets
    tf = _cifar_transforms()
    train = datasets.CIFAR10(root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root, train=False, download=True, transform=tf)
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader, train


def tinyimagenet_loaders(
    root: str, batch_size: int = BATCH_SIZE, num_workers: int = 4, image_size: int = 224,
    augment_train: bool = True,
):
    from torchvision import datasets
    train_tf = _tinyimagenet_transforms(train=augment_train, image_size=image_size)
    eval_tf = _tinyimagenet_transforms(train=False, image_size=image_size)
    train = datasets.ImageFolder(f"{root}/train", transform=train_tf)
    val = datasets.ImageFolder(f"{root}/val", transform=eval_tf)
    train_loader = DataLoader(train, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train


def calibration_loader(
    train_dataset,
    calib_size: int,
    batch_size: int = BATCH_SIZE,
    seed: Optional[int] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Random subset of the training set used to build the channel ranks.

    A fresh random subset is drawn per rank (paper: "Every rank is created with a
    random subset of the training data"). Pass a seed for reproducibility.

    num_workers defaults to 0, not >0. ranking.rank_channels calls evaluate_accuracy
    on this SAME loader hundreds of times (once per candidate channel, Algorithm 3's
    O(L*C)), and each call iterates the loader fresh — with num_workers>0 that means
    spawning new worker processes every single time (slow, especially on Windows),
    which dwarfs the actual forward-pass cost since the calibration subset is small
    (2500-5000 images) and max_batches caps each call to just a few batches anyway.
    Measured ~4.4x faster at num_workers=0 for this repeated-fresh-iteration pattern.
    """
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    n = len(train_dataset)
    calib_size = min(calib_size, n)
    idx = torch.randperm(n, generator=g)[:calib_size].tolist()
    subset = Subset(train_dataset, idx)
    return DataLoader(subset, batch_size, shuffle=False, num_workers=num_workers)
