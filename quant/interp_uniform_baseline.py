"""
interp_uniform_baseline.py — derive the matched-budget interpolated uniform baseline
from already-logged Direct-quantization accuracies (results.csv), instead of
doing the arithmetic by hand. Reads the most recent Direct 3/4/5-bit rows for
the given experiment and linearly interpolates to the requested budget(s).

Usage:
    python -m quant.interp_uniform_baseline --experiment mobilenetv2_cifar10 --r 0.40
    python -m quant.interp_uniform_baseline --experiment vit_b16_tinyimagenet --r 0.40
"""
import argparse
import csv
import json


def load_direct_accuracies(csv_path: str, experiment: str) -> dict[int, float]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["experiment"] == experiment and row["method"] == "direct" and row["section"] == "table2":
                rows.append(row)
    if not rows:
        raise ValueError(f"no direct/table2 rows found for {experiment} in {csv_path}")
    # keep the most recent timestamp per bit level (results.csv is append-only)
    latest: dict[int, tuple[str, float]] = {}
    for row in rows:
        bits = int(row["bits"])
        ts = row["timestamp"]
        acc = float(row["accuracy"])
        if bits not in latest or ts > latest[bits][0]:
            latest[bits] = (ts, acc)
    return {b: v[1] for b, v in latest.items()}


def interp(direct: dict[int, float], budget: float) -> float:
    lo, hi = int(budget), int(budget) + 1
    if budget == lo:
        return direct[lo]
    if lo not in direct or hi not in direct:
        raise ValueError(f"missing direct accuracy at {lo} or {hi} bits for budget {budget}")
    frac = budget - lo
    return direct[lo] + frac * (direct[hi] - direct[lo])


def main(experiment: str, r: float, ms=(1, 2, 3), n_bits: int = 3,
          csv_path: str = "quant/experiment_logs/results.csv"):
    direct = load_direct_accuracies(csv_path, experiment)
    print(f"logged Direct accuracies for {experiment}: "
          + ", ".join(f"{b}b={a*100:.2f}%" for b, a in sorted(direct.items())))

    out = {}
    for m in ms:
        budget = n_bits + r * m
        acc = interp(direct, budget)
        out[m] = dict(budget=budget, uniform_accuracy=acc)
        print(f"m={m}: budget={budget:.2f} -> interpolated uniform = {acc*100:.2f}%")

    out_path = f"quant/experiment_logs/table44_uniform_{experiment}.json"
    with open(out_path, "w") as f:
        json.dump({"experiment": experiment, "r": r, "n_bits": n_bits, "results": out}, f, indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--r", type=float, required=True)
    args = ap.parse_args()
    main(args.experiment, args.r)
