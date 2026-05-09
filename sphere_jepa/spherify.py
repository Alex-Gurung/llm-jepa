from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def spherify(
    z: torch.Tensor,
    radius: float | None = None,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Project vectors to the hypersphere with RMS-norm radius sqrt(d)."""

    dim = z.shape[-1]
    target_radius = math.sqrt(dim) if radius is None else radius
    return F.normalize(z, dim=-1, eps=eps) * target_radius


def perturb_and_respherify(
    v: torch.Tensor,
    sigma_max: float = 1.0,
    sigma: torch.Tensor | float | None = None,
    *,
    radius: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add isotropic Gaussian noise and project back to the sphere.

    When ``sigma_max == 0.0`` and no explicit ``sigma`` is supplied, the
    function returns ``v`` unchanged. That exact path is the sigma=0 control
    used throughout the experiments.
    """

    if sigma_max == 0.0 and sigma is None:
        return v, torch.zeros(v.shape[:-1], device=v.device, dtype=v.dtype).unsqueeze(-1)

    if sigma is None:
        sigma_tensor = torch.rand(v.shape[:-1], device=v.device, dtype=v.dtype).unsqueeze(-1)
        sigma_tensor = sigma_tensor * sigma_max
    elif isinstance(sigma, torch.Tensor):
        sigma_tensor = sigma.to(device=v.device, dtype=v.dtype)
        if sigma_tensor.ndim == v.ndim - 1:
            sigma_tensor = sigma_tensor.unsqueeze(-1)
    else:
        sigma_tensor = torch.full(v.shape[:-1] + (1,), float(sigma), device=v.device, dtype=v.dtype)

    noisy = v + sigma_tensor * torch.randn_like(v)
    return spherify(noisy, radius=radius), sigma_tensor
