"""
diagnose_mobilenetv2_greedy_variance.py — post-hoc diagnostic for the
MobileNetV2 greedy-ranking seed-to-seed variance flagged in Table 4.1 (note *3,
std=2.29pp across 3 seeds: 86.22% / 85.93% / 82.11%) and discussed as
"unexplained" in dissertation §5.2.1.

This does NOT rerun greedy (expensive, O(L*C), ~84-154 min/seed on this model).
It reuses the three cached per-seed rank files already produced by that run
(ranks_mobilenetv2_cifar10_3bit_fullscan[.json / _calibseed1.json /
_calibseed2.json], seeds 0/1/2) and asks: how much do the three seeds actually
agree, layer by layer, on (a) which single channel is "most important"
(P[layer], eDQA's Algorithm 3 line 14) and (b) the broader top-round(r*C)
protected-channel SET that actually determines downstream accuracy?

Method:
  - For each layer, compare P[layer] across the 3 seeds (exact match / not).
  - For each layer, compute the pairwise Jaccard overlap of the top-round(r*C)
    channel sets across the 3 seed pairs, and average.
  - Compare that observed Jaccard against the EXPECTED Jaccard of two
    independent random k-of-C subsets (chance baseline): for k = round(r*C),
    E[|A∩B|] ≈ k²/C, E[|A∪B|] ≈ 2k − k²/C, chance_Jaccard ≈ ratio of these.
    This is a standard first-moment approximation (not the true expectation of
    a ratio), used here only to flag which layers are ~indistinguishable from
    a random channel choice given the calibration noise, not as an exact
    combinatorial figure.
  - Layers whose observed Jaccard is within 0.05 of the chance baseline are
    flagged as "noise-dominated": the 512-image calibration signal (Table 3.1:
    max_batches=4, batch_size=128) used inside Algorithm 3's accuracy
    evaluation is not distinguishing candidate channels any better than a coin
    flip at that layer.

Usage:
    python -m quant.diagnose_mobilenetv2_greedy_variance
"""
import json
import math
import statistics

RANK_FILES = {
    0: "ranks_mobilenetv2_cifar10_3bit_fullscan.json",
    1: "ranks_mobilenetv2_cifar10_3bit_fullscan_calibseed1.json",
    2: "ranks_mobilenetv2_cifar10_3bit_fullscan_calibseed2.json",
}
R = 0.40  # MobileNetV2's protection ratio


def topk(rank_list, C, r=R):
    k = round(C * r)
    return set(rank_list[:k])


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a or b) else 1.0


def chance_jaccard(C, k):
    overlap = k * k / C
    union = 2 * k - overlap
    return overlap / union if union > 0 else 1.0


def main():
    ranks = {s: json.load(open(f)) for s, f in RANK_FILES.items()}
    layers = list(ranks[0].keys())

    rows = []
    p1_agree_all = 0
    for layer in layers:
        r0, r1, r2 = ranks[0][layer], ranks[1][layer], ranks[2][layer]
        C = len(r0)
        k = round(C * R)
        p0, p1, p2 = r0[0], r1[0], r2[0]
        if p0 == p1 == p2:
            p1_agree_all += 1
        k0, k1, k2 = topk(r0, C), topk(r1, C), topk(r2, C)
        j01, j02, j12 = jaccard(k0, k1), jaccard(k0, k2), jaccard(k1, k2)
        mean_j = statistics.mean([j01, j02, j12])
        chance = chance_jaccard(C, k)
        rows.append(dict(layer=layer, C=C, k=k, top1_agree=(p0 == p1 == p2),
                          mean_jaccard=mean_j, chance_jaccard=chance,
                          excess=mean_j - chance))

    near_chance = [r for r in rows if r["excess"] < 0.05]
    above_chance = [r for r in rows if r["excess"] >= 0.05]

    xs = [r["C"] for r in rows]
    ys = [r["excess"] for r in rows]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    pearson_r = cov / (sx * sy) if sx * sy > 0 else None

    summary = dict(
        num_layers=len(rows),
        top1_agree_all_seeds=p1_agree_all,
        top1_disagree_at_least_one_seed=len(rows) - p1_agree_all,
        near_chance_layers=len(near_chance),
        near_chance_layer_names=[r["layer"] for r in near_chance],
        mean_C_near_chance=statistics.mean(r["C"] for r in near_chance) if near_chance else None,
        mean_C_above_chance=statistics.mean(r["C"] for r in above_chance) if above_chance else None,
        overall_mean_observed_jaccard=statistics.mean(r["mean_jaccard"] for r in rows),
        overall_mean_chance_jaccard=statistics.mean(r["chance_jaccard"] for r in rows),
        pearson_corr_C_vs_excess=pearson_r,
        rows=rows,
    )

    print(f"layers: {summary['num_layers']}")
    print(f"top-1 pick agrees across all 3 seeds: {summary['top1_agree_all_seeds']}/{summary['num_layers']}")
    print(f"layers at/near chance-level protected-set overlap (excess<0.05): "
          f"{summary['near_chance_layers']}/{summary['num_layers']}")
    print(f"  -> {summary['near_chance_layer_names']}")
    print(f"mean channel count, near-chance layers:   {summary['mean_C_near_chance']:.1f}")
    print(f"mean channel count, above-chance layers:  {summary['mean_C_above_chance']:.1f}")
    print(f"overall mean observed Jaccard (protected set): {summary['overall_mean_observed_jaccard']:.3f}")
    print(f"overall mean chance-baseline Jaccard:          {summary['overall_mean_chance_jaccard']:.3f}")
    print(f"Pearson r(channel count, excess-over-chance):  {summary['pearson_corr_C_vs_excess']:.3f}")

    out_path = "quant/experiment_logs/diagnose_mobilenetv2_greedy_variance.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
