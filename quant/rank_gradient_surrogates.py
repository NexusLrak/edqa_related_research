"""
rank_gradient_surrogates.py
=================================================

Gradient-based channel-importance criteria: saliency, the (p, q) family
(energy/saliency/fisher/fisher_a2), and the protection_gain-based family
(pg / pg^2 / eDQA-saliency = pg*|g|).

RECONSTRUCTION NOTE (2026-09-01): the original scripts that produced these
criteria (`pg_gradient_combos.py`, `nm_sweep_pg_gradient.py`,
`boundary_n2_n4_mbv2_vit.py`, and others referenced by name in RESULTS.md /
EXTENSION_WORK.md) were lost before this project's code was packaged for
submission -- confirmed absent from quant/, quant/diagnostics/, model/,
packaging/, and remote_backup/. This module reimplements them from the
formulas and cost profile documented in RESULTS.md Section E/F/12 and
EXTENSION_WORK.md sections 9-12, and from the one surviving fragment,
model/kaggle_kernel/kaggle_vit_energy_saliency_blend.py (a ViT-only, pre-
protection_gain Kaggle kernel whose retain_grad()/backward() pattern --
including the memory-fragmentation fix below -- is reused here unchanged).
Validated against the reference-configuration (n=3, m=3) numbers already
published for ResNet-32 in RESULTS.md; single-seed re-runs are not expected
to reproduce those figures bit-for-bit, only within noise. See
packaging/template/README.md and EXTENSION_WORK.md for the disclosure note.

Design idea
-----------
Every criterion here is a special case of one of two families, both built
from the SAME single calibration forward(+backward) pass per layer:

  1. The (p, q) family over (activation, gradient):
         C_pq(A) = sum_{i,j} |A_ij|^p * |G_ij|^q
     energy=(2,0), saliency=(1,1), fisher=(0,2), fisher_a2=(2,2) -- see
     `compute_pq_scores` / `rank_channels_via_pq`.

  2. The same (p, q) construction applied to protection_gain (a per-channel
     scalar, not per-element) in place of activation magnitude:
         C'_pq = pg^p * (sum_ij |G_ij|)^q
     pg^2=(2,0) (energy's role, no gradient needed), eDQA-saliency=(1,1)
     (saliency's role) -- see `protection_gain_scores` / `rank_channels_via_pg2`
     / `rank_channels_via_edqa_saliency`.

Typical usage
-------------
    scores = compute_pq_scores(model, calib, layer_names, channel_dim, device)
    ranks = rank_channels_via_pq(scores, p=1, q=1)   # saliency

    pg = protection_gain_scores(model, calib, layer_names, channel_dim, device,
                                 n_bits=3, m=3)
    ranks = rank_channels_via_pg2(pg)                # pg^2, no backward pass
"""

from __future__ import annotations

import gc

import torch

from .quantizer import fake_quantize_direct


def _channel_sum(t: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Sum a per-sample activation/gradient tensor down to one scalar per
    channel, regardless of how many non-channel (spatial/token) dims it has.

    Same idea as rank_surrogates.py's helpers: move the channel axis to dim 1,
    then reduce every other dim. Handles CNN 4D (B,C,H,W), ViT-block 3D
    (B,N_tokens,C), and 2D (B,C) activations (e.g. a final Linear/fc).
    """
    if channel_dim != 1:
        t = t.movedim(channel_dim, 1)
    if t.dim() == 4:
        return t.sum(dim=(0, 2, 3))
    if t.dim() == 3:
        return t.sum(dim=(0, 2))
    if t.dim() == 2:
        return t.sum(dim=0)
    raise ValueError(f"unexpected activation ndim {t.dim()}")


@torch.no_grad()
def _channel_sum_sq_error(x: torch.Tensor, n_bits: int, channel_dim: int,
                          clip_percentile: float | None) -> torch.Tensor:
    """Per-channel sum of squared quantization error at `n_bits`, used by
    protection_gain_scores. One extra fake_quantize_direct call each.
    """
    recon = fake_quantize_direct(x, n_bits, channel_dim=channel_dim, clip_percentile=clip_percentile)
    err2 = (x - recon).pow(2)
    return _channel_sum(err2, channel_dim)


def compute_pq_scores(
    model,
    calib_loader,
    layer_names,
    channel_dim: "int | dict[str, int]" = 1,
    device: str = "cpu",
    calib_batches: int = 1,
) -> dict[str, dict[str, torch.Tensor]]:
    """One calibration forward+backward pass, accumulating every raw
    per-channel sum needed for the whole (p, q) family at once:

        a2    = sum |a|^2         -- energy,     (p,q)=(2,0)
        ag    = sum |a| * |g|     -- saliency,    (p,q)=(1,1)
        g2    = sum |g|^2         -- fisher,      (p,q)=(0,2)
        a2g2  = sum |a|^2 * |g|^2 -- fisher_a2,   (p,q)=(2,2)
        g_abs = sum |g|           -- gradient term for eDQA-saliency (pg * |g|)

    Uses register_forward_hook + Tensor.retain_grad() (not a full backward
    hook) so the activation itself, not just its grad, is directly available
    -- same approach as model/kaggle_kernel/kaggle_vit_energy_saliency_blend.py.

    IMPORTANT: after backward(), the retain_grad() graph must be explicitly
    freed (gc.collect() + empty_cache() + synchronize()) before any downstream
    eval pass, or GPU memory fragmentation silently degrades it by ~6x. This
    was found the hard way in the original exploration (see
    EXTENSION_WORK.md's "retain_grad碎片化bug" note) -- do not remove.

    Returns {layer_name: {"a2": Tensor[C], "ag": ..., "g2": ..., "a2g2": ...,
    "g_abs": ...}}.
    """
    import torch.nn.functional as F

    modules = dict(model.named_modules())
    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            t.retain_grad()
            activations[name] = t
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    keys = ("a2", "ag", "g2", "a2g2", "g_abs")
    accum: dict[str, dict[str, "torch.Tensor | None"]] = {
        name: {k: None for k in keys} for name in layer_names
    }

    for i, (x, y) in enumerate(calib_loader):
        if i >= calib_batches:
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
                d = 1
            else:
                d = dim

            vals = {
                "a2": _channel_sum(a.pow(2), d),
                "ag": _channel_sum(a.abs() * g.abs(), d),
                "g2": _channel_sum(g.pow(2), d),
                "a2g2": _channel_sum(a.pow(2) * g.pow(2), d),
                "g_abs": _channel_sum(g.abs(), d),
            }
            for k in keys:
                v = vals[k].cpu()
                accum[name][k] = v if accum[name][k] is None else accum[name][k] + v

    for h in handles:
        h.remove()

    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.startswith("cuda") if isinstance(device, str) else False:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return accum


_PQ_KEY = {(2, 0): "a2", (1, 1): "ag", (0, 2): "g2", (2, 2): "a2g2"}


def rank_channels_via_pq(
    scores: dict[str, dict[str, torch.Tensor]], p: int, q: int
) -> dict[str, list[int]]:
    """Select one member of the (p, q) family from `compute_pq_scores`'s
    output and turn it into a ranks dict (higher score = more important,
    channel 0 in the list is most important).

    (p, q) must be one of (2,0) energy, (1,1) saliency, (0,2) fisher,
    (2,2) fisher_a2 -- the four points evaluated in the dissertation's
    (p, q) grid sweep (Table 3.8).
    """
    if (p, q) not in _PQ_KEY:
        raise ValueError(f"unsupported (p,q)={(p, q)}; choose from {list(_PQ_KEY)}")
    key = _PQ_KEY[(p, q)]
    return {
        name: layer_scores[key].argsort(descending=True).tolist()
        for name, layer_scores in scores.items()
    }


def rank_channels_via_saliency(model, calib_loader, layer_names, device,
                               channel_dim: "int | dict[str, int]" = 1,
                               calib_batches: int = 1) -> dict[str, list[int]]:
    scores = compute_pq_scores(model, calib_loader, layer_names, channel_dim, device, calib_batches)
    return rank_channels_via_pq(scores, p=1, q=1)


def rank_channels_via_fisher(model, calib_loader, layer_names, device,
                             channel_dim: "int | dict[str, int]" = 1,
                             calib_batches: int = 1) -> dict[str, list[int]]:
    scores = compute_pq_scores(model, calib_loader, layer_names, channel_dim, device, calib_batches)
    return rank_channels_via_pq(scores, p=0, q=2)


def rank_channels_via_fisher_a2(model, calib_loader, layer_names, device,
                                channel_dim: "int | dict[str, int]" = 1,
                                calib_batches: int = 1) -> dict[str, list[int]]:
    scores = compute_pq_scores(model, calib_loader, layer_names, channel_dim, device, calib_batches)
    return rank_channels_via_pq(scores, p=2, q=2)


def protection_gain_scores(
    model,
    calib_loader,
    layer_names,
    channel_dim: "int | dict[str, int]" = 1,
    device: str = "cpu",
    n_bits: int = 3,
    m: int = 3,
    clip_percentile: float | None = None,
    calib_batches: int = 1,
) -> dict[str, torch.Tensor]:
    """protection_gain = distortion_N - distortion_{N+m}, per channel.

    distortion_k is the summed squared reconstruction error of a channel's
    calibration activations after fake_quantize_direct at k bits -- an exact
    measurement (not an approximation), matching the dissertation's §3.2.3
    definition and its "computed with the same fake_quantize_direct routine"
    claim. No backward pass needed.
    """
    modules = dict(model.named_modules())
    activations: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_m, _i, o):
            t = o[0] if isinstance(o, tuple) else o
            activations[name] = t.detach()
        return hook

    for name in layer_names:
        handles.append(modules[name].register_forward_hook(make_hook(name)))

    pg: dict[str, "torch.Tensor | None"] = {name: None for name in layer_names}

    with torch.no_grad():
        for i, (x, _y) in enumerate(calib_loader):
            if i >= calib_batches:
                break
            model(x.to(device))
            for name in layer_names:
                act = activations[name]
                dim = channel_dim[name] if isinstance(channel_dim, dict) else channel_dim
                dist_n = _channel_sum_sq_error(act, n_bits, dim, clip_percentile)
                dist_nm = _channel_sum_sq_error(act, n_bits + m, dim, clip_percentile)
                delta = (dist_n - dist_nm).cpu()
                pg[name] = delta if pg[name] is None else pg[name] + delta

    for h in handles:
        h.remove()

    return pg


def rank_channels_via_pg2(pg_scores: dict[str, torch.Tensor]) -> dict[str, list[int]]:
    """pg^2 -- the protection_gain-family analogue of energy, (p,q)=(2,0) in
    C'_pq = pg^p * |g|^q. Squaring a non-negative quantity is order-preserving,
    so this ranks identically to plain `pg_scores` (a useful sanity check, and
    also why the reference-config accuracy matches pure protection_gain almost
    exactly -- see RESULTS.md's ResNet-32 row: both 87.79%).
    """
    return {name: v.pow(2).argsort(descending=True).tolist() for name, v in pg_scores.items()}


def rank_channels_via_edqa_saliency(
    model,
    calib_loader,
    layer_names,
    channel_dim: "int | dict[str, int]" = 1,
    device: str = "cpu",
    n_bits: int = 3,
    m: int = 3,
    clip_percentile: float | None = None,
    calib_batches: int = 1,
) -> dict[str, list[int]]:
    """eDQA-saliency = pg * |g|, this project's proposed substitute for
    eDQA's greedy Algorithm 3 (dissertation §3.2.3). Cost: one forward and
    one backward pass (for |g|, via compute_pq_scores) plus two additional
    fake_quantize_direct calls per calibration batch (for pg, via
    protection_gain_scores) -- matches the cost profile documented in §3.2.3
    and Figure 3.2.
    """
    pg = protection_gain_scores(
        model, calib_loader, layer_names, channel_dim, device, n_bits, m, clip_percentile, calib_batches,
    )
    pq = compute_pq_scores(model, calib_loader, layer_names, channel_dim, device, calib_batches)
    return {
        name: (pg[name] * pq[name]["g_abs"]).argsort(descending=True).tolist()
        for name in layer_names
    }
