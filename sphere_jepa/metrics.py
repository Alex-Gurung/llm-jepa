from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _singular_values(z: torch.Tensor) -> torch.Tensor:
    """Return singular values via the smaller Gram matrix when possible."""

    z = z.float()
    if z.numel() == 0 or min(z.shape) == 0:
        return torch.empty(0, dtype=z.dtype, device=z.device)
    if z.shape[0] <= z.shape[1]:
        gram = z @ z.T
    else:
        gram = z.T @ z
    vals = torch.linalg.eigvalsh(gram).clamp_min(0.0).sqrt()
    return vals.flip(0)


def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    z = z - z.mean(dim=0, keepdim=True)
    if z.numel() == 0 or min(z.shape) == 0:
        return 0.0
    s = _singular_values(z)
    p = s / (s.sum() + eps)
    p = p.clamp_min(eps)
    return torch.exp(-(p * p.log()).sum()).item()


def within_class_rankme(z: torch.Tensor, labels: torch.Tensor) -> dict:
    """Per-class RankMe plus a mean over classes with at least two samples."""

    out: dict[int | str, float] = {}
    values = []
    for c in labels.unique().tolist():
        mask = labels == c
        if int(mask.sum().item()) > 1:
            value = rankme(z[mask])
            out[int(c)] = value
            values.append(value)
    out["_mean_within_class"] = float(sum(values) / len(values)) if values else 0.0
    return out


def uniformity_loss(z: torch.Tensor, t: float = 2.0, max_pairs: int = 50_000) -> float:
    """Wang-Isola uniformity with pair subsampling for large batches."""

    z = F.normalize(z, dim=-1)
    n = len(z)
    if n < 2:
        return math.nan
    if n * (n - 1) // 2 > max_pairs:
        sample_n = max(2, int((2 * max_pairs) ** 0.5))
        idx = torch.randperm(n, device=z.device)[:sample_n]
        z = z[idx]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log().item()


def alignment_loss(z_a: torch.Tensor, z_b: torch.Tensor) -> float:
    z_a, z_b = F.normalize(z_a, dim=-1), F.normalize(z_b, dim=-1)
    return (z_a - z_b).pow(2).sum(dim=-1).mean().item()


def _subsample_for_pairwise(z: torch.Tensor, max_pairs: int) -> torch.Tensor:
    n = len(z)
    if n * (n - 1) // 2 <= max_pairs:
        return z
    sample_n = max(2, int((2 * max_pairs) ** 0.5))
    idx = torch.arange(sample_n, device=z.device) * n // sample_n
    return z[idx]


def cosine_pair_stats(z: torch.Tensor, max_pairs: int = 50_000) -> dict:
    z = F.normalize(z, dim=-1)
    if len(z) < 2:
        return {"mean": math.nan, "std": math.nan, "p95": math.nan}
    z = _subsample_for_pairwise(z, max_pairs)
    c = z @ z.T
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    c_off = c[mask]
    return {
        "mean": c_off.mean().item(),
        "std": c_off.std().item(),
        "p95": c_off.quantile(0.95).item(),
    }


def cosine_histogram(z: torch.Tensor, bins: int = 40, max_pairs: int = 50_000) -> dict:
    z = F.normalize(z, dim=-1)
    if len(z) < 2:
        return {"bin_edges": [], "counts": []}
    z = _subsample_for_pairwise(z, max_pairs)
    c = z @ z.T
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    c_off = c[mask].float().cpu()
    hist = torch.histogram(c_off, bins=bins, range=(-1.0, 1.0))
    return {
        "bin_edges": [float(x) for x in hist.bin_edges.tolist()],
        "counts": [int(x) for x in hist.hist.tolist()],
    }


def eigen_spectrum(z: torch.Tensor, top_k: int = 32) -> dict:
    z = z.detach().float()
    if z.numel() == 0 or min(z.shape) == 0:
        return {"mass": [], "cumulative": []}
    centered = z - z.mean(dim=0, keepdim=True)
    s = _singular_values(centered)
    mass = s.pow(2)
    total = mass.sum()
    if total <= 0:
        values = torch.zeros_like(mass[:top_k])
    else:
        values = mass[:top_k] / total
    cumulative = values.cumsum(dim=0)
    return {
        "mass": [float(x) for x in values.tolist()],
        "cumulative": [float(x) for x in cumulative.tolist()],
    }


def pca_scatter(
    z: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    max_points: int = 800,
) -> list[dict]:
    z = z.detach().float().cpu()
    if z.ndim != 2 or len(z) == 0:
        return []
    n = min(len(z), max_points)
    idx = torch.linspace(0, len(z) - 1, steps=n).round().long()
    sampled = z[idx]
    centered = sampled - sampled.mean(dim=0, keepdim=True)
    if centered.shape[1] == 1:
        coords = torch.cat([centered, torch.zeros_like(centered)], dim=1)
    else:
        q = min(2, centered.shape[0], centered.shape[1])
        _, _, components = torch.pca_lowrank(centered, q=q, center=False, niter=3)
        coords = centered @ components[:, :q]
        if coords.shape[1] == 1:
            coords = torch.cat([coords, torch.zeros_like(coords)], dim=1)
    out = []
    labels_cpu = labels.detach().cpu() if labels is not None else None
    for point_idx, coord in zip(idx.tolist(), coords):
        row = {
            "x": float(coord[0]),
            "y": float(coord[1]),
            "index": int(point_idx),
        }
        if labels_cpu is not None:
            row["label"] = int(labels_cpu[point_idx])
        out.append(row)
    return out


def embedding_visual_diagnostics(
    z: torch.Tensor,
    labels: torch.Tensor | None = None,
    *,
    max_points: int = 800,
    hist_bins: int = 40,
    spectrum_k: int = 32,
) -> dict:
    return {
        "pca_scatter": pca_scatter(z, labels, max_points=max_points),
        "cosine_histogram": cosine_histogram(z, bins=hist_bins),
        "eigen_spectrum": eigen_spectrum(z, top_k=spectrum_k),
    }


def anchor_geometry_diagnostics(anchor: torch.Tensor) -> dict:
    """Diagnostics that must be reported before cap-anchor training."""

    anchor = anchor.detach()
    centered = anchor - anchor.mean(dim=0, keepdim=True)
    if min(centered.shape) == 0:
        top_ratio = 0.0
    else:
        s = _singular_values(centered)
        mass = s.pow(2).sum()
        top_ratio = (s[0].pow(2) / mass).item() if mass > 0 else 0.0
    a_normed = F.normalize(anchor, dim=-1)
    return {
        "rankme": rankme(anchor),
        "top_eig_mass_ratio": top_ratio,
        "mean_norm": anchor.norm(dim=-1).mean().item(),
        "uniformity_after_normalize": uniformity_loss(a_normed),
        "cosine_stats": cosine_pair_stats(anchor),
        "cosine_histogram": cosine_histogram(anchor),
        "eigen_spectrum": eigen_spectrum(anchor),
    }


def paired_vs_unpaired_cosine(z_a: torch.Tensor, z_b: torch.Tensor) -> dict:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    sim = z_a @ z_b.T
    paired = sim.diag()
    mask = ~torch.eye(sim.shape[0], dtype=torch.bool, device=sim.device)
    unpaired = sim[mask]
    return {
        "paired_mean": paired.mean().item(),
        "paired_std": paired.std().item() if len(paired) > 1 else 0.0,
        "unpaired_mean": unpaired.mean().item() if len(unpaired) else math.nan,
        "unpaired_p95": unpaired.quantile(0.95).item() if len(unpaired) else math.nan,
    }


def retrieval_at_k(query: torch.Tensor, target: torch.Tensor, ks: tuple[int, ...] = (1, 10)) -> dict:
    query = F.normalize(query, dim=-1)
    target = F.normalize(target, dim=-1)
    sim = query @ target.T
    order = sim.argsort(dim=-1, descending=True)
    labels = torch.arange(query.shape[0], device=query.device).unsqueeze(1)
    out = {}
    for k in ks:
        out[f"recall@{k}"] = (order[:, :k] == labels).any(dim=1).float().mean().item()
    out["mean_rank"] = ((order == labels).nonzero()[:, 1].float() + 1.0).mean().item()
    return out


def similarity_heatmap(query: torch.Tensor, target: torch.Tensor, *, max_items: int = 48) -> dict:
    """Small cosine matrix for visual inspection of paired retrieval structure."""

    n = min(len(query), len(target), max_items)
    if n == 0:
        return {"values": [], "paired": []}
    q = F.normalize(query[:n].detach().float().cpu(), dim=-1)
    t = F.normalize(target[:n].detach().float().cpu(), dim=-1)
    sim = q @ t.T
    return {
        "values": [[float(v) for v in row] for row in sim.tolist()],
        "paired": [float(v) for v in sim.diag().tolist()],
    }
