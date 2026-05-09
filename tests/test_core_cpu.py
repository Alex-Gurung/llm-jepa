import math

import torch
import torch.nn as nn

from sphere_jepa.heads import MLP
from sphere_jepa.losses import clean_jepa_alignment, noisy_cap_anchor_loss
from sphere_jepa.metrics import anchor_geometry_diagnostics, uniformity_loss, within_class_rankme
from sphere_jepa.spherify import perturb_and_respherify, spherify


def test_spherify_uses_sqrt_dim_radius():
    z = torch.randn(8, 16)
    v = spherify(z)
    assert torch.allclose(v.norm(dim=-1), torch.full((8,), 4.0), atol=1e-5)


def test_zero_sigma_returns_input_unchanged():
    v = spherify(torch.randn(4, 5))
    out, sigma = perturb_and_respherify(v, sigma_max=0.0)
    assert torch.equal(out, v)
    assert torch.equal(sigma, torch.zeros(4, 1))


def test_losses_are_differentiable():
    z_a = torch.randn(6, 8, requires_grad=True)
    z_b = torch.randn(6, 8)
    predictor = MLP(8, 8)
    cap = MLP(8, 3)
    anchor = torch.randn(6, 3)
    loss = clean_jepa_alignment(z_a, z_b, predictor)
    loss = loss + noisy_cap_anchor_loss(z_a, anchor, cap, sigma_max=0.1)
    loss.backward()
    assert z_a.grad is not None
    assert math.isfinite(float(loss.detach()))


def test_metrics_handle_labels_and_large_uniformity_subsample():
    z = torch.randn(128, 12)
    labels = torch.arange(128) % 4
    wc = within_class_rankme(z, labels)
    assert "_mean_within_class" in wc
    assert math.isfinite(uniformity_loss(z, max_pairs=100))
    diagnostics = anchor_geometry_diagnostics(z)
    assert "top_eig_mass_ratio" in diagnostics
    assert diagnostics["top_eig_mass_ratio"] >= 0
