"""
build_package.py
=================================================

Assembles edqa_portable_package.zip: run.py + quant/ + vendored ResNet-32/
MobileNetV2 architecture defs + all 4 checkpoints + the CIFAR-10/TinyImageNet
archives, ready to unzip and run on any rented GPU box (see
packaging/template/README.md, which ships inside the zip).

Run this LOCALLY (needs access to this repo's model/checkpoints/, data/, and
the local torch.hub cache for cifar10_resnet32/cifar10_mobilenetv2_x1_0's
pretrained weights) -- it is not itself part of the portable package.

Usage:
    python packaging/build_package.py
    python packaging/build_package.py --experiments resnet18_tinyimagenet,vit_b16_tinyimagenet
    python packaging/build_package.py --skip-data   # code+checkpoints only, mount data separately
"""

from __future__ import annotations

import argparse
import os
import zipfile

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "template")
TORCH_HUB_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub")

# experiment -> (checkpoint src path, checkpoint arcname, data files needed [(src, arcname), ...])
CIFAR10_DATA = [(os.path.join(PROJ_ROOT, "data", "cifar-10-python.tar.gz"), "data/cifar-10-python.tar.gz")]
TINYIMAGENET_DATA = [(os.path.join(PROJ_ROOT, "data", "tiny-imagenet-200.zip"), "data/tiny-imagenet-200.zip")]

EXPERIMENTS = {
    "resnet32_cifar10": dict(
        checkpoint=(
            os.path.join(TORCH_HUB_DIR, "checkpoints", "cifar10_resnet32-ef93fc4d.pt"),
            "checkpoints/cifar10_resnet32.pt",
        ),
        data=CIFAR10_DATA,
        needs_vendor=True,
    ),
    "mobilenetv2_cifar10": dict(
        checkpoint=(
            os.path.join(TORCH_HUB_DIR, "checkpoints", "cifar10_mobilenetv2_x1_0-fe6a5b48.pt"),
            "checkpoints/cifar10_mobilenetv2_x1_0.pt",
        ),
        data=CIFAR10_DATA,
        needs_vendor=True,
    ),
    "resnet18_tinyimagenet": dict(
        checkpoint=(
            os.path.join(PROJ_ROOT, "model", "checkpoints", "resnet18_tinyimagenet.pt"),
            "checkpoints/resnet18_tinyimagenet.pt",
        ),
        data=TINYIMAGENET_DATA,
        needs_vendor=False,
    ),
    "vit_b16_tinyimagenet": dict(
        checkpoint=(
            os.path.join(PROJ_ROOT, "model", "checkpoints", "vit_b16_tinyimagenet.pt"),
            "checkpoints/vit_b16_tinyimagenet.pt",
        ),
        data=TINYIMAGENET_DATA,
        needs_vendor=False,
    ),
}

# only these two need the vendored architecture defs (no torch.hub network call)
VENDOR_SRC = os.path.join(TORCH_HUB_DIR, "chenyaofo_pytorch-cifar-models_master", "pytorch_cifar_models")


def _add_file(zf: zipfile.ZipFile, src: str, arcname: str, compress: bool = True):
    if not os.path.exists(src):
        raise FileNotFoundError(f"expected {src} (needed for {arcname}) -- see README for what to fetch first")
    zf.write(src, arcname, compress_type=zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED)


def _add_dir(zf: zipfile.ZipFile, src_dir: str, arc_prefix: str, exclude_dirs=("__pycache__", "tests")):
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for fname in files:
            if fname.endswith(".pyc"):
                continue
            src = os.path.join(root, fname)
            rel = os.path.relpath(src, src_dir)
            zf.write(src, os.path.join(arc_prefix, rel).replace(os.sep, "/"), compress_type=zipfile.ZIP_DEFLATED)


def build(output_path: str, experiments: list[str], include_data: bool):
    unknown = set(experiments) - set(EXPERIMENTS)
    if unknown:
        raise ValueError(f"unknown experiment(s) {unknown}, choose from {list(EXPERIMENTS)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    seen_data_arcnames: set[str] = set()
    needs_vendor = False

    with zipfile.ZipFile(output_path, "w") as zf:
        # -- template files (run.py, README.md, requirements.txt, run.ipynb) --
        for fname in os.listdir(TEMPLATE_DIR):
            src = os.path.join(TEMPLATE_DIR, fname)
            if fname == "README.md" and set(experiments) != set(EXPERIMENTS):
                with open(src, encoding="utf-8") as f:
                    text = f.read()
                banner = (
                    f"> **本包只打包了 `{', '.join(experiments)}`**（不是全部4个模型）——"
                    f"`run.py --experiment` 只有这{len(experiments)}个可用，下面命令里其他模型的例子请忽略。\n\n"
                )
                zf.writestr(fname, banner + text)
                continue
            _add_file(zf, src, fname)

        # -- quant/ package, top-level .py files only --
        quant_dir = os.path.join(PROJ_ROOT, "quant")
        for fname in sorted(os.listdir(quant_dir)):
            if fname.endswith(".py"):
                _add_file(zf, os.path.join(quant_dir, fname), f"quant/{fname}")

        # -- per-experiment checkpoint + data --
        for name in experiments:
            cfg = EXPERIMENTS[name]
            ckpt_src, ckpt_arc = cfg["checkpoint"]
            _add_file(zf, ckpt_src, ckpt_arc, compress=False)
            needs_vendor = needs_vendor or cfg["needs_vendor"]
            if include_data:
                for data_src, data_arc in cfg["data"]:
                    if data_arc in seen_data_arcnames:
                        continue
                    _add_file(zf, data_src, data_arc, compress=False)
                    seen_data_arcnames.add(data_arc)

        # -- vendored resnet32/mobilenetv2 architecture defs, only if needed --
        if needs_vendor:
            if not os.path.isdir(VENDOR_SRC):
                raise FileNotFoundError(
                    f"{VENDOR_SRC} not found -- run torch.hub.load('chenyaofo/pytorch-cifar-models', ...) "
                    "once locally first so this cache directory gets populated"
                )
            _add_dir(zf, VENDOR_SRC, "vendor/pytorch_cifar_models")

        # empty output/ dir placeholder so it exists after unzip
        zf.writestr("output/.gitkeep", "")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"wrote {output_path} ({size_mb:.0f} MB), experiments={experiments}, include_data={include_data}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS), help="comma-separated subset of experiments")
    ap.add_argument(
        "--output", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dist", "edqa_portable_package.zip"),
    )
    ap.add_argument("--skip-data", action="store_true", help="omit data/ (code+checkpoints only, ~few hundred MB smaller)")
    args = ap.parse_args()
    build(args.output, args.experiments.split(","), include_data=not args.skip_data)


if __name__ == "__main__":
    main()
