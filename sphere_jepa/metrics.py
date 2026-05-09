from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    z = z - z.mean(dim=0, keepdim=True)
    if z.numel() == 0 or min(z.shape) == 0:
        return 0.0
    s = torch.linalg.svdvals(z.float())
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


def cosine_pair_stats(z: torch.Tensor) -> dict:
    z = F.normalize(z, dim=-1)
    if len(z) < 2:
        return {"mean": math.nan, "std": math.nan, "p95": math.nan}
    c = z @ z.T
    mask = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    c_off = c[mask]
    return {
        "mean": c_off.mean().item(),
        "std": c_off.std().item(),
        "p95": c_off.quantile(0.95).item(),
    }


def anchor_geometry_diagnostics(anchor: torch.Tensor) -> dict:
    """Diagnostics that must be reported before cap-anchor training."""

    anchor = anchor.detach()
    centered = anchor - anchor.mean(dim=0, keepdim=True)
    if min(centered.shape) == 0:
        top_ratio = 0.0
    else:
        s = torch.linalg.svdvals(centered.float())
        mass = s.pow(2).sum()
        top_ratio = (s[0].pow(2) / mass).item() if mass > 0 else 0.0
    a_normed = F.normalize(anchor, dim=-1)
    return {
        "rankme": rankme(anchor),
        "top_eig_mass_ratio": top_ratio,
        "mean_norm": anchor.norm(dim=-1).mean().item(),
        "uniformity_after_normalize": uniformity_loss(a_normed),
        "cosine_stats": cosine_pair_stats(anchor),
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
