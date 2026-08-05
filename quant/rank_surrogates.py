"""
rank_surrogates.py

Channel-importance scoring via (effective) matrix-rank surrogates, intended as a
drop-in alternative to eDQA's greedy channel-importance ranking.

Design idea
-----------
For each channel we build a matrix from its calibration activations, take its
singular-value spectrum ONCE, and derive every rank surrogate from that single
spectrum. Trying many surrogates is therefore nearly free.

Two decisions, both natural ablation axes:
  1. How to build the per-channel matrix   -> `mode`
  2. Which scalar surrogate to read off it  -> `metric`

Related work worth reading: HRank (Lin et al., CVPR 2020) uses the average rank
of feature maps as a filter-importance criterion for pruning -- close prior art
for the "rank as importance" hypothesis, just applied to a different task.

Typical usage
-------------
    acts = collect_activations(layer, calib_loader)   # [N, C, H, W]
    scores = channel_rank_scores(acts, mode="sample_spatial", metric="stable_rank")
    order = torch.argsort(scores, descending=True)    # channels, most important first

    # feasibility check: how close is this ordering to the greedy one?
    print(spearman(scores, greedy_scores))
"""

from __future__ import annotations
import torch


def spectrum_metrics(sv: torch.Tensor, rtol: float = 1e-3,
                     energy_thr: float = 0.99) -> dict:
    """Derive all rank surrogates from a batch of singular-value spectra.

    Parameters
    ----------
    sv : Tensor of shape [B, k]
        Singular values per matrix, sorted descending, non-negative.
    rtol : relative threshold for numerical rank (tol = rtol * sigma_max).
    energy_thr : cumulative-energy fraction for the energy-threshold rank.

    Returns
    -------
    dict[str, Tensor[B]] : one scalar per input matrix, for each surrogate.
    """
    sv = sv.clamp_min(0.0)
    eps = 1e-12
    smax = sv[:, :1]                       # [B, 1]
    s2 = sv.pow(2)
    total = sv.sum(dim=1)                  # [B]
    total2 = s2.sum(dim=1)                 # [B]

    # 1. Numerical rank: count singular values above a relative threshold.
    numerical_rank = (sv > rtol * smax).sum(dim=1)

    # 2. Stable rank: ||A||_F^2 / ||A||_2^2  -- cheap, continuous, noise-robust.
    stable_rank = total2 / (smax.squeeze(1).pow(2) + eps)

    # 3. Effective rank: exp(entropy of normalized singular values) (Roy & Vetterli, 2007).
    p = sv / (total.unsqueeze(1) + eps)
    entropy = -(p * (p + eps).log()).sum(dim=1)
    effective_rank = entropy.exp()

    # 4. Energy-threshold rank: #components needed to reach `energy_thr` of the energy.
    cum_energy = s2.cumsum(dim=1) / (total2.unsqueeze(1) + eps)
    energy_rank = (cum_energy < energy_thr).sum(dim=1) + 1

    # 5. Nuclear norm (Schatten-1): sum of singular values.
    nuclear_norm = total

    # 6. Participation ratio: (sum s^2)^2 / sum s^4 -- effective dimensionality.
    participation_ratio = total2.pow(2) / (s2.pow(2).sum(dim=1) + eps)

    return {
        "numerical_rank": numerical_rank.float(),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "energy_rank": energy_rank.float(),
        "nuclear_norm": nuclear_norm,
        "participation_ratio": participation_ratio,
    }


METRICS = ("numerical_rank", "stable_rank", "effective_rank",
           "energy_rank", "nuclear_norm", "participation_ratio")


@torch.no_grad()
def channel_rank_scores(activations: torch.Tensor,
                        mode: str = "sample_spatial",
                        metric: str = "stable_rank",
                        center: bool = False,
                        rtol: float = 1e-3,
                        energy_thr: float = 0.99) -> torch.Tensor:
    """Score every channel of a conv activation tensor (higher = more important).

    Parameters
    ----------
    activations : Tensor [N, C, H, W] -- a batch of calibration activations.
    mode :
        "sample_spatial" - one [N, H*W] matrix per channel; rank ~ how diverse
                           the channel's response is across samples. One SVD per channel.
        "feature_map"    - rank of each [H, W] feature map, averaged over the N
                           samples (HRank-style). One SVD per (sample, channel).
    metric : which key of `spectrum_metrics` to return.
    center : subtract the per-matrix mean before SVD (covariance-style rank).

    Returns
    -------
    Tensor [C] of importance scores.
    """
    if activations.dim() != 4:
        raise ValueError("expected activations of shape [N, C, H, W]")
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {METRICS}")
    N, C, H, W = activations.shape
    a = activations.float()

    if mode == "sample_spatial":
        m = a.permute(1, 0, 2, 3).reshape(C, N, H * W)      # [C, N, HW]
        if center:
            m = m - m.mean(dim=2, keepdim=True)
        sv = torch.linalg.svdvals(m)                        # [C, min(N, HW)]
        return spectrum_metrics(sv, rtol, energy_thr)[metric]

    if mode == "feature_map":
        m = a.reshape(N * C, H, W)                          # [N*C, H, W]
        if center:
            m = m - m.mean(dim=(1, 2), keepdim=True)
        sv = torch.linalg.svdvals(m)                        # [N*C, min(H, W)]
        per = spectrum_metrics(sv, rtol, energy_thr)[metric]  # [N*C]
        return per.reshape(N, C).mean(dim=0)                # [C]

    raise ValueError(f"unknown mode {mode!r}; use 'sample_spatial' or 'feature_map'")


@torch.no_grad()
def all_channel_scores(activations: torch.Tensor,
                       mode: str = "sample_spatial",
                       center: bool = False,
                       rtol: float = 1e-3,
                       energy_thr: float = 0.99) -> dict:
    """Compute every surrogate at once (single SVD, all metrics)."""
    if activations.dim() != 4:
        raise ValueError("expected activations of shape [N, C, H, W]")
    N, C, H, W = activations.shape
    a = activations.float()
    if mode == "sample_spatial":
        m = a.permute(1, 0, 2, 3).reshape(C, N, H * W)
        if center:
            m = m - m.mean(dim=2, keepdim=True)
        sv = torch.linalg.svdvals(m)
        return spectrum_metrics(sv, rtol, energy_thr)
    if mode == "feature_map":
        m = a.reshape(N * C, H, W)
        if center:
            m = m - m.mean(dim=(1, 2), keepdim=True)
        sv = torch.linalg.svdvals(m)
        return {k: v.reshape(N, C).mean(dim=0)
                for k, v in spectrum_metrics(sv, rtol, energy_thr).items()}
    raise ValueError(f"unknown mode {mode!r}")


@torch.no_grad()
def rank_channels_via_surrogate_per_layer(
    model,
    calib_loader,
    test_loader,
    layer_names,
    device,
    metric: str = "stable_rank",
    channel_dim: "int | dict[str, int]" = 1,
    clip_percentile: float | None = None,
    n_bits: int = 3,
    m: int = 3,
    r: float = 0.55,
    calib_batches: int = 1,
    eval_batches: int = 8,
    margin_threshold: float = 0.0,
    verbose: bool = False,
):
    """Cheap substitute for Algorithm 3's greedy search (ranking.rank_channels):
    rank every layer's channels from a single SVD pass, then -- since a single
    global direction (low-rank-first) doesn't hold uniformly across layers, see
    quant/experiment_logs/2026-07-16_rank_surrogate_full_network.log -- pick
    each layer's own best direction (ascending vs descending) via a cheap
    isolated single-layer eval (that layer quantized with eDQA, every other
    layer at full precision).

    Cost: O(L) SVD passes + O(2L) isolated eDQA evals, vs Algorithm 3's O(L*C)
    full-model evals (C up to 512 for these layers) -- independent of channel
    count, so the gap widens for wider layers.

    `clip_percentile` is threaded through the direction-search evals too, so the
    chosen direction is the one that's actually best under whatever scale
    variant (see run_experiments_tuned.VARIANTS) will be used downstream.

    `margin_threshold`: the isolated low-vs-high eval (n ~= eval_batches * batch
    size) is noisy -- at eval_batches=8 the single-proportion standard error is
    ~1.5pp, and quant/experiment_logs/2026-07-16_rank_surrogate_per_layer_direction.log
    showed 19/21 layers had a low-vs-high margin under 2pp, i.e. within noise of
    a coin flip (only conv1 had a clear ~25pp margin). When
    abs(accs['high'] - accs['low']) < margin_threshold for a layer, that
    direction pick is untrustworthy, so this falls back to a magnitude/energy
    ranking for that layer instead: per-channel sum-of-squared activation
    (descending -- unambiguous "high energy = important", same family as
    L1-norm filter-pruning criteria, no direction search needed). Default 0.0
    disables the cascade (always trusts the rank-direction winner, matching
    prior behaviour).

    Returns {layer_name: [channel_ids most-important-first]}, a drop-in ranks
    dict for compare_methods / sweep_ratio / sweep_extra_bits.
    """
    # Deferred imports: avoids a hard dependency on hooks/ranking/compression
    # for callers that only want the pure scoring functions above.
    from torch.utils.data import DataLoader

    from .compression import get_compressor
    from .hooks import QuantManager
    from .ranking import evaluate_accuracy

    # The direction search below calls evaluate_accuracy on `test_loader`
    # 2 * (# layers with a direction choice) times, each a fresh iteration.
    # If test_loader has num_workers>0 (cifar10_loaders/tinyimagenet_loaders
    # default to 4), Windows respawns worker processes on every fresh
    # iteration -- the same repeated-fresh-iteration cost that data.
    # calibration_loader was already fixed for (see its docstring), just
    # hitting test_loader here instead. Measured 2026-07-18: 746.9s for 33
    # layers (66 evals) on ResNet-32/CIFAR-10 with num_workers=4, ~11.3s/eval
    # -- almost entirely worker-spawn overhead, not compute (this model/image
    # size is small). Build a num_workers=0 loader over the same dataset just
    # for this loop; the caller's test_loader (used once, start to finish, for
    # the real accuracy eval elsewhere) is untouched.
    direction_search_loader = DataLoader(
        test_loader.dataset, batch_size=test_loader.batch_size, shuffle=False, num_workers=0,
    )

    modules = dict(model.named_modules())
    act_holders = {name: {} for name in layer_names}
    handles = []
    for name in layer_names:
        def make_hook(nm):
            def hook(_m, _i, o):
                # torchvision's MultiheadAttention returns (attn_output, attn_weights),
                # not a bare Tensor -- see hooks.vit_full_target_layers' docstring.
                t = o[0] if isinstance(o, tuple) else o
                act_holders[nm]["x"] = t.detach().cpu()
            return hook
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    acts_per_layer = {name: [] for name in layer_names}
    for i, (x, _y) in enumerate(calib_loader):
        if i >= calib_batches:
            break
        model(x.to(device))
        for name in layer_names:
            acts_per_layer[name].append(act_holders[name]["x"])
    for h in handles:
        h.remove()

    scores_per_layer = {}
    energy_per_layer = {}
    for name in layer_names:
        acts = torch.cat(acts_per_layer[name], dim=0)
        dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
        if dim != 1:
            acts = acts.movedim(dim, 1)
        if acts.dim() == 4:
            scores_per_layer[name] = (all_channel_scores(acts, mode="sample_spatial")[metric], True)
            energy_per_layer[name] = acts.float().pow(2).sum(dim=(0, 2, 3))
        elif acts.dim() == 2:
            # No spatial dim (e.g. a final Linear/fc): SVD-rank is degenerate
            # (every channel is trivially "rank 1"). Magnitude-range fallback.
            scores_per_layer[name] = (acts.abs().amax(dim=0) - acts.abs().amin(dim=0), False)
        else:
            raise ValueError(f"layer {name}: unexpected activation ndim {acts.dim()}")

    comp = get_compressor("identity")
    ranks = {}
    for name in layer_names:
        scores, has_direction_choice = scores_per_layer[name]
        if not has_direction_choice:
            ranks[name] = scores.argsort(descending=True).tolist()
            continue

        asc_rank = scores.argsort(descending=False).tolist()
        desc_rank = scores.argsort(descending=True).tolist()
        accs = {}
        dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
        for direction, rank_list in (("low", asc_rank), ("high", desc_rank)):
            with QuantManager(model, [name], channel_dim=dim, clip_percentile=clip_percentile) as mgr:
                mgr.set_edqa(name, n_bits, m, r, rank_list, comp)
                accs[direction] = evaluate_accuracy(model, direction_search_loader, device, eval_batches)
        margin = abs(accs["high"] - accs["low"])

        if margin < margin_threshold:
            ranks[name] = energy_per_layer[name].argsort(descending=True).tolist()
            source = f"energy-fallback (margin={margin*100:.2f}pp < {margin_threshold*100:.2f}pp)"
        else:
            winner = max(accs, key=accs.get)
            ranks[name] = asc_rank if winner == "low" else desc_rank
            source = f"{winner}-rank-first (margin={margin*100:.2f}pp)"

        if verbose:
            print(f"  {name:30s} low={accs['low']*100:6.2f}%  high={accs['high']*100:6.2f}%  -> {source}")

    return ranks


@torch.no_grad()
def rank_channels_via_energy(
    model,
    calib_loader,
    layer_names,
    device,
    channel_dim: "int | dict[str, int]" = 1,
    calib_batches: int = 1,
):
    """Control for rank_channels_via_surrogate_per_layer's margin_threshold
    cascade: ranks EVERY layer by per-channel energy (sum-of-squared
    activation, descending), no rank surrogate, no direction search, no
    isolated per-layer eval. Disentangles "the margin cascade helps" from
    "energy is just a better importance signal than rank here regardless of
    margin" -- if this scores close to the cascade's result, the margin
    gating isn't the active ingredient.

    Returns {layer_name: [channel_ids most-important-first]}.
    """
    modules = dict(model.named_modules())
    act_holders = {name: {} for name in layer_names}
    handles = []
    for name in layer_names:
        def make_hook(nm):
            def hook(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                act_holders[nm]["x"] = t.detach().cpu()
            return hook
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    acts_per_layer = {name: [] for name in layer_names}
    for i, (x, _y) in enumerate(calib_loader):
        if i >= calib_batches:
            break
        model(x.to(device))
        for name in layer_names:
            acts_per_layer[name].append(act_holders[name]["x"])
    for h in handles:
        h.remove()

    ranks = {}
    for name in layer_names:
        acts = torch.cat(acts_per_layer[name], dim=0)
        dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
        if dim != 1:
            acts = acts.movedim(dim, 1)
        if acts.dim() == 4:
            energy = acts.float().pow(2).sum(dim=(0, 2, 3))
        elif acts.dim() == 3:
            # ViT MLPBlock output (B, N_tokens, hidden_dim) after movedim -> (B, hidden_dim, N_tokens):
            # same idea as the 4D conv case, just one "spatial" axis (tokens) instead of two (H, W).
            energy = acts.float().pow(2).sum(dim=(0, 2))
        elif acts.dim() == 2:
            energy = acts.float().pow(2).sum(dim=0)
        else:
            raise ValueError(f"layer {name}: unexpected activation ndim {acts.dim()}")
        ranks[name] = energy.argsort(descending=True).tolist()

    return ranks


@torch.no_grad()
def rank_channels_via_stratified(
    model,
    calib_loader,
    layer_names,
    device,
    r: float,
    n_bins: int = 4,
    channel_dim: "int | dict[str, int]" = 1,
    calib_batches: int = 1,
):
    """2026-07-19: bin channels by matrix-rank score into `n_bins` equal-sized groups
    (ascending rank), then within EACH bin independently take the top r-fraction by
    energy. Every bin contributes the same r-fraction, so -- unlike rank_channels_via_
    surrogate_per_layer -- there's no ascending/descending direction choice to make,
    sidesteps the per-layer margin-noise problem found there (most layers' direction
    signal was within noise, see rank_energy_cascade.py). Motivation: pure energy
    selection could be systematically starving certain rank-strata of protection if
    rank and energy correlate; stratifying by rank first guarantees every stratum gets
    representation, chosen (within that stratum) by energy.

    Cost ~= pure energy (one calibration forward pass + a cheap SVD per layer, no
    isolated per-layer eval) -- `n_bins` only affects a CPU-side sort/bin step on
    already-computed per-channel scores, doesn't change ranking cost.

    2D layers (no spatial dim) fall back to the same magnitude-range ranking as
    rank_channels_via_surrogate_per_layer, unstratified (rank is degenerate there).

    Returns {layer_name: [channel_ids most-important-first]}.
    """
    modules = dict(model.named_modules())
    act_holders = {name: {} for name in layer_names}
    handles = []
    for name in layer_names:
        def make_hook(nm):
            def hook(_m, _i, o):
                t = o[0] if isinstance(o, tuple) else o
                act_holders[nm]["x"] = t.detach().cpu()
            return hook
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    acts_per_layer = {name: [] for name in layer_names}
    for i, (x, _y) in enumerate(calib_loader):
        if i >= calib_batches:
            break
        model(x.to(device))
        for name in layer_names:
            acts_per_layer[name].append(act_holders[name]["x"])
    for h in handles:
        h.remove()

    ranks = {}
    for name in layer_names:
        acts = torch.cat(acts_per_layer[name], dim=0)
        dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
        if dim != 1:
            acts = acts.movedim(dim, 1)

        if acts.dim() == 2:
            magnitude = acts.abs().amax(dim=0) - acts.abs().amin(dim=0)
            ranks[name] = magnitude.argsort(descending=True).tolist()
            continue
        if acts.dim() != 4:
            raise ValueError(f"layer {name}: unexpected activation ndim {acts.dim()}")

        rank_score = all_channel_scores(acts, mode="sample_spatial")["stable_rank"]
        energy_score = acts.float().pow(2).sum(dim=(0, 2, 3))
        C = acts.shape[1]

        order_by_rank = rank_score.argsort(descending=False)  # low-rank bin first
        bin_size = -(-C // n_bins)  # ceil division
        important, rest = [], []
        for b in range(n_bins):
            bin_channels = order_by_rank[b * bin_size: (b + 1) * bin_size]
            if bin_channels.numel() == 0:
                continue
            k = max(1, round(bin_channels.numel() * r))
            local_order = energy_score[bin_channels].argsort(descending=True)
            sorted_bin = bin_channels[local_order]
            important.extend(sorted_bin[:k].tolist())
            rest.extend(sorted_bin[k:].tolist())
        ranks[name] = important + rest

    return ranks


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation between two channel-score vectors (numpy only)."""
    import numpy as np
    x = a.detach().cpu().numpy().ravel()
    y = b.detach().cpu().numpy().ravel()
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = (np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum())) + 1e-12
    return float((rx * ry).sum() / denom)


if __name__ == "__main__":
    # Sanity demo: first 8 channels are rank-1 across samples (a fixed spatial
    # pattern scaled per sample -> low effective rank); the rest are random.
    torch.manual_seed(0)
    N, C, H, W = 64, 32, 12, 12
    acts = torch.randn(N, C, H, W)
    pattern = torch.randn(H, W)
    for c in range(8):
        coeff = torch.randn(N, 1, 1)
        acts[:, c] = coeff * pattern          # each sample = scalar * same pattern

    scores = all_channel_scores(acts, mode="sample_spatial")
    print(f"{'metric':20s}  {'low-rank ch (0-7)':>18s}  {'random ch (8-31)':>18s}")
    for name, s in scores.items():
        lo = s[:8].mean().item()
        hi = s[8:].mean().item()
        print(f"{name:20s}  {lo:18.3f}  {hi:18.3f}")
