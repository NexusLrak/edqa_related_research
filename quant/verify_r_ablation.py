"""
verify_r_ablation.py — multi-seed version of the ResNet-18 protected-ratio (r)
ablation (RESULTS.md/EXTENSION_WORK.md: "把ResNet-18的r临时改成0.40重测（单seed
ablation）..."). The single-seed version narrowed the energy-vs-saliency gap from
-1.81pp (r=0.55, 3-seed baseline at the time) to -0.76pp (r=0.40, single seed) --
"too thin an experiment... to call a real effect" per the dissertation's own
§5.2.1/§4.7.1 caveat. This reruns energy AND saliency at r=0.40 across 3 seeds,
matching the standard multi-seed threshold used elsewhere in this project.

NOTE on the baseline to compare against: the r=0.55 (ResNet-18's own default) gap
has since been updated to a confirmed 5-seed value of -2.23pp (RESULTS.md line 199,
"5-seed: -2.23pp, 比3-seed时的-1.81pp差距还略微扩大了") -- the dissertation's current
text still cites the superseded 3-seed figure (-1.81pp) as the "before" comparison.
This script does not need to re-measure r=0.55 (already 5-seed confirmed); it only
adds seeds to the r=0.40 side, but the two numbers should be compared against the
CURRENT r=0.55 baseline (-2.23pp), not the stale one, when updating the dissertation.

Usage:
    python -m quant.verify_r_ablation
"""
import json
import time

import torch

from quant.data import calibration_loader
from quant.rank_gradient_surrogates import rank_channels_via_saliency
from quant.rank_surrogates import rank_channels_via_energy
from quant.ranking import evaluate_accuracy
from quant.hooks import QuantManager
from quant.compression import get_compressor
from quant.run_experiments import EXPERIMENTS

EXPERIMENT = "resnet18_tinyimagenet"
R_ABLATION = 0.40
N_BITS = 3
M = 3
CLIP_PERCENTILE = 99.9
SEEDS = (0, 1, 2)


def eval_ranking(model, ranks, layer_names, channel_dim, test_loader, device):
    comp = get_compressor("identity")
    with QuantManager(model, layer_names, channel_dim=channel_dim, clip_percentile=CLIP_PERCENTILE) as mgr:
        for ln in layer_names:
            if ln in ranks:
                mgr.set_edqa(ln, N_BITS, M, R_ABLATION, ranks[ln], comp)
        return evaluate_accuracy(model, test_loader, device=device, max_batches=None)


def main():
    cfg = EXPERIMENTS[EXPERIMENT]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, r_ablation={R_ABLATION} (default r={cfg['r']})")

    model = cfg["build"]().to(device)
    train_loader, test_loader, train_set = cfg["loaders"]()
    layer_names = cfg["target_layers"](model)
    channel_dim = cfg["channel_dim"]
    if isinstance(layer_names, tuple):
        layer_names, channel_dim = layer_names

    energy_accs, saliency_accs = [], []
    for seed in SEEDS:
        calib = calibration_loader(train_set, cfg["calib_size"], seed=seed)
        t0 = time.time()
        e_ranks = rank_channels_via_energy(model, calib, layer_names, device, channel_dim=channel_dim)
        e_acc = eval_ranking(model, e_ranks, layer_names, channel_dim, test_loader, device)
        print(f"seed={seed} energy:   {e_acc*100:.2f}%  ({time.time()-t0:.1f}s)")
        energy_accs.append(e_acc)

        calib = calibration_loader(train_set, cfg["calib_size"], seed=seed)
        t0 = time.time()
        s_ranks = rank_channels_via_saliency(model, calib, layer_names, device, channel_dim=channel_dim)
        s_acc = eval_ranking(model, s_ranks, layer_names, channel_dim, test_loader, device)
        print(f"seed={seed} saliency: {s_acc*100:.2f}%  ({time.time()-t0:.1f}s)")
        saliency_accs.append(s_acc)

    import statistics
    e_mean, s_mean = statistics.mean(energy_accs), statistics.mean(saliency_accs)
    e_std = statistics.pstdev(energy_accs) if len(energy_accs) > 1 else 0.0
    s_std = statistics.pstdev(saliency_accs) if len(saliency_accs) > 1 else 0.0
    gap = (s_mean - e_mean) * 100

    print(f"\nenergy   3-seed mean={e_mean*100:.2f}% std={e_std*100:.2f}pp  ({[f'{a*100:.2f}' for a in energy_accs]})")
    print(f"saliency 3-seed mean={s_mean*100:.2f}% std={s_std*100:.2f}pp  ({[f'{a*100:.2f}' for a in saliency_accs]})")
    print(f"gap (saliency - energy) at r={R_ABLATION}: {gap:+.2f}pp")

    out = dict(experiment=EXPERIMENT, r=R_ABLATION, n_bits=N_BITS, m=M, seeds=list(SEEDS),
               energy=dict(accs=energy_accs, mean=e_mean, std=e_std),
               saliency=dict(accs=saliency_accs, mean=s_mean, std=s_std),
               gap_pp=gap)
    out_path = "quant/experiment_logs/r_ablation_resnet18.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
