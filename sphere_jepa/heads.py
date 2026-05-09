from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spherify import spherify


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        hidden_dim = hidden_dim or max(in_dim, out_dim) * 2
        layers: list[nn.Module] = []
        cur = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    return module


def make_ema_copy(module: nn.Module) -> nn.Module:
    return freeze_module(copy.deepcopy(module))


@torch.no_grad()
def update_ema(ema_module: nn.Module, module: nn.Module, momentum: float = 0.99) -> None:
    for ema_param, param in zip(ema_module.parameters(), module.parameters()):
        ema_param.data.mul_(momentum).add_(param.detach().data, alpha=1.0 - momentum)
    for ema_buffer, buffer in zip(ema_module.buffers(), module.buffers()):
        ema_buffer.copy_(buffer)


@dataclass
class AnchorPreprocessor:
    mode: str = "raw"
    eps: float = 1e-5
    mean: torch.Tensor | None = None
    components: torch.Tensor | None = None
    scale: torch.Tensor | None = None

    def fit(self, anchors: torch.Tensor) -> "AnchorPreprocessor":
        if self.mode not in {"raw", "norm", "white", "sphere"}:
            raise ValueError(f"unknown anchor preprocessing mode: {self.mode}")
        if self.mode == "white":
            x = anchors.float()
            self.mean = x.mean(dim=0, keepdim=True)
            centered = x - self.mean
            _, s, vh = torch.linalg.svd(centered, full_matrices=False)
            denom = max(1, x.shape[0] - 1)
            eig = s.pow(2) / denom
            self.components = vh
            self.scale = torch.rsqrt(eig + self.eps)
        return self

    def transform(self, anchors: torch.Tensor) -> torch.Tensor:
        if self.mode == "raw":
            return anchors
        if self.mode == "norm":
            return F.normalize(anchors, dim=-1)
        if self.mode == "sphere":
            return spherify(anchors)
        if self.mode == "white":
            if self.mean is None or self.components is None or self.scale is None:
                raise RuntimeError("white preprocessor must be fit before transform")
            x = (anchors.float() - self.mean.to(anchors.device))
            x = x @ self.components.to(anchors.device).T
            x = x * self.scale.to(anchors.device)
            return x.to(dtype=anchors.dtype)
        raise ValueError(f"unknown anchor preprocessing mode: {self.mode}")

    def fit_transform(self, anchors: torch.Tensor) -> torch.Tensor:
        return self.fit(anchors).transform(anchors)
