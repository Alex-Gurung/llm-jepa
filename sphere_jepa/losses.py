from __future__ import annotations

import torch
import torch.nn.functional as F

from .spherify import perturb_and_respherify, spherify


def clean_jepa_alignment(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    predictor: torch.nn.Module,
    *,
    spherify_inputs: bool = True,
    detach_target: bool = True,
    metric: str = "cosine",
) -> torch.Tensor:
    """Standard JEPA alignment: predict ``z_b`` from ``z_a``.

    The default metric is cosine, matching the LLM-JEPA setup. MSE and L2 are
    kept as ablations.
    """

    if spherify_inputs:
        z_a, z_b = spherify(z_a), spherify(z_b)
    target = z_b.detach() if detach_target else z_b
    pred = predictor(z_a)
    if metric == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    if metric == "mse":
        return F.mse_loss(pred, target)
    if metric == "l2":
        return torch.linalg.norm(pred - target, ord=2, dim=-1).mean()
    raise ValueError(f"unknown metric: {metric}")


def noisy_cap_anchor_loss(
    z: torch.Tensor,
    anchor: torch.Tensor,
    cap_predictor: torch.nn.Module,
    *,
    sigma_max: float = 0.5,
    sphere_radius: float | None = None,
    detach_anchor: bool = True,
) -> torch.Tensor:
    """Own-view noisy cap reconstruction against an external anchor."""

    v = spherify(z, radius=sphere_radius)
    v_noisy, _ = perturb_and_respherify(v, sigma_max=sigma_max, radius=sphere_radius)
    target = anchor.detach() if detach_anchor else anchor
    return F.mse_loss(cap_predictor(v_noisy), target)


def cross_view_cap_anchor_loss(
    z_a: torch.Tensor,
    anchor_b: torch.Tensor,
    cap_predictor: torch.nn.Module,
    *,
    sigma_max: float = 0.5,
    sphere_radius: float | None = None,
    detach_anchor: bool = True,
) -> torch.Tensor:
    """Noisy cap from view A predicts paired-view anchor B."""

    v_a = spherify(z_a, radius=sphere_radius)
    v_a_noisy, _ = perturb_and_respherify(v_a, sigma_max=sigma_max, radius=sphere_radius)
    target = anchor_b.detach() if detach_anchor else anchor_b
    return F.mse_loss(cap_predictor(v_a_noisy), target)


def noisy_cap_token_loss(
    z: torch.Tensor,
    target_token_ids: torch.Tensor,
    cap_predictor: torch.nn.Module,
    *,
    sigma_max: float = 0.5,
    sphere_radius: float | None = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Token-anchor variant: noisy spherical cap predicts tokens directly."""

    v = spherify(z, radius=sphere_radius)
    v_noisy, _ = perturb_and_respherify(v, sigma_max=sigma_max, radius=sphere_radius)
    logits = cap_predictor(v_noisy)
    return F.cross_entropy(
        logits.flatten(0, 1),
        target_token_ids.flatten(),
        ignore_index=ignore_index,
    )


def combined_loss(
    jepa_term: torch.Tensor,
    cap_term: torch.Tensor,
    *,
    lambda_jepa: float = 1.0,
    lambda_cap: float = 1.0,
) -> torch.Tensor:
    return lambda_jepa * jepa_term + lambda_cap * cap_term


def info_nce_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    z_a = F.normalize(z_a, dim=-1)
    z_b = F.normalize(z_b, dim=-1)
    logits = z_a @ z_b.T / temperature
    labels = torch.arange(z_a.shape[0], device=z_a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("expected a square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    *,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg objective for paired embeddings."""

    repr_loss = F.mse_loss(z_a, z_b)
    std_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_b = torch.sqrt(z_b.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1.0 - std_a)) + torch.mean(F.relu(1.0 - std_b))

    z_a = z_a - z_a.mean(dim=0)
    z_b = z_b - z_b.mean(dim=0)
    denom = max(1, z_a.shape[0] - 1)
    cov_a = (z_a.T @ z_a) / denom
    cov_b = (z_b.T @ z_b) / denom
    cov_loss = _off_diagonal(cov_a).pow(2).sum() / z_a.shape[-1]
    cov_loss = cov_loss + _off_diagonal(cov_b).pow(2).sum() / z_b.shape[-1]
    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss


def sigreg_loss(
    z: torch.Tensor,
    *,
    mean_coeff: float = 1.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """LeJEPA-style isotropic Gaussian regularizer for one batch."""

    z = z.float()
    z_centered = z - z.mean(dim=0, keepdim=True)
    mean_loss = z.mean(dim=0).pow(2).mean()
    denom = max(1, z.shape[0] - 1)
    cov = (z_centered.T @ z_centered) / denom
    eye = torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    cov_loss = (cov - eye).pow(2).mean()
    std = torch.sqrt(z.var(dim=0) + eps)
    std_floor = F.relu(1.0 - std).mean()
    return mean_coeff * (mean_loss + std_floor) + cov_coeff * cov_loss
