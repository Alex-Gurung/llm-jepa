from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sphere_jepa.heads import MLP, make_ema_copy, update_ema
from sphere_jepa.losses import (
    clean_jepa_alignment,
    noisy_cap_anchor_loss,
    sigreg_loss,
    vicreg_loss,
)
from sphere_jepa.metrics import (
    alignment_loss,
    anchor_geometry_diagnostics,
    cosine_pair_stats,
    rankme,
    uniformity_loss,
    within_class_rankme,
)
from sphere_jepa.spherify import perturb_and_respherify, spherify


VARIANTS = [
    "A",
    "B",
    "C",
    "C_detach",
    "C_ema",
    "D0a",
    "D1a",
    "D0b",
    "D1b",
    "D0c",
    "D1c",
    "D3",
    "F",
    "G",
]


@dataclass
class ToyConfig:
    seed: int = 0
    clusters: int = 8
    samples_per_cluster: int = 128
    emb_dim: int = 16
    hidden_dim: int = 64
    rff_dim: int = 32
    batch_size: int = 256
    steps: int = 250
    teacher_steps: int = 250
    lr: float = 2e-3
    teacher_lr: float = 2e-3
    sigma_max: float = 0.5
    lambda_jepa: float = 1.0
    lambda_cap: float = 1.0
    ema_momentum: float = 0.99


class Encoder(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IdentityTeacher(nn.Module):
    def __init__(self, feature_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, feature_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(x))


class RFFAnchor:
    def __init__(self, in_dim: int, out_dim: int, *, generator: torch.Generator) -> None:
        self.weight = torch.randn(in_dim, out_dim, generator=generator) * 2.0
        self.bias = torch.rand(out_dim, generator=generator) * (2.0 * math.pi)
        self.scale = math.sqrt(2.0 / out_dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(x @ self.weight.to(x.device) + self.bias.to(x.device))


def set_seed(seed: int) -> torch.Generator:
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def make_toy_data(config: ToyConfig, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    angles = torch.linspace(0, 2 * math.pi, config.clusters + 1)[:-1]
    centers = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1) * 4.0
    points = []
    labels = []
    for idx, center in enumerate(centers):
        noise = torch.randn(config.samples_per_cluster, 2, generator=generator) * 0.35
        points.append(center.unsqueeze(0) + noise)
        labels.extend([idx] * config.samples_per_cluster)
    x = torch.cat(points, dim=0)
    y = torch.tensor(labels, dtype=torch.long)
    perm = torch.randperm(len(x), generator=generator)
    return x[perm], y[perm]


def make_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    def one_view() -> torch.Tensor:
        theta = (torch.rand((), device=x.device) - 0.5) * 0.35
        scale = 0.9 + 0.2 * torch.rand((), device=x.device)
        rot = torch.tensor(
            [[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]],
            device=x.device,
            dtype=x.dtype,
        )
        jitter = torch.randn_like(x) * 0.12
        bias = torch.randn(1, 2, device=x.device, dtype=x.dtype) * 0.05
        return scale * (x @ rot.T) + jitter + bias

    return one_view(), one_view()


def pretrain_teacher(x: torch.Tensor, config: ToyConfig) -> IdentityTeacher:
    teacher = IdentityTeacher(feature_dim=config.rff_dim, hidden_dim=config.hidden_dim)
    opt = torch.optim.AdamW(teacher.parameters(), lr=config.teacher_lr)
    for step in range(config.teacher_steps):
        idx = torch.randint(0, len(x), (min(config.batch_size, len(x)),))
        xb = x[idx]
        recon = teacher(xb)
        loss = F.mse_loss(recon, xb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def get_anchor(
    kind: str,
    x: torch.Tensor,
    teacher: IdentityTeacher,
    rff: RFFAnchor,
) -> torch.Tensor:
    if kind == "continuous":
        return x
    if kind == "rff":
        return rff(x)
    if kind == "teacher":
        return teacher.features(x)
    raise ValueError(f"unknown anchor kind: {kind}")


def train_variant(
    variant: str,
    x: torch.Tensor,
    labels: torch.Tensor,
    config: ToyConfig,
    teacher: IdentityTeacher,
    rff: RFFAnchor,
) -> dict:
    encoder = Encoder(config.emb_dim, config.hidden_dim)
    predictor = MLP(config.emb_dim, config.emb_dim, hidden_dim=config.hidden_dim, num_layers=2)
    cap_dim = config.emb_dim
    if variant.endswith("a"):
        cap_dim = 2
    elif variant.endswith("b") or variant.endswith("c"):
        cap_dim = config.rff_dim
    cap_predictor = MLP(config.emb_dim, cap_dim, hidden_dim=config.hidden_dim, num_layers=2)
    classifier = MLP(config.emb_dim, config.clusters, hidden_dim=config.hidden_dim, num_layers=2)
    ema_encoder = make_ema_copy(encoder) if variant == "C_ema" else None

    params = list(encoder.parameters())
    if variant not in {"F"}:
        params.extend(predictor.parameters())
    if variant.startswith("C") or variant.startswith("D"):
        params.extend(cap_predictor.parameters())
    if variant == "D3":
        params.extend(classifier.parameters())
    opt = torch.optim.AdamW(params, lr=config.lr, weight_decay=1e-4)

    history = []
    for step in range(config.steps):
        idx = torch.randint(0, len(x), (min(config.batch_size, len(x)),))
        xb = x[idx]
        yb = labels[idx]
        view_a, view_b = make_views(xb)
        z_a = encoder(view_a)
        z_b = encoder(view_b)

        if variant == "A":
            loss = F.mse_loss(predictor(z_a), z_b.detach())
        elif variant == "B":
            loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
        elif variant == "C":
            target = spherify(z_a)
            noisy, _ = perturb_and_respherify(target, sigma_max=config.sigma_max)
            loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            loss = loss + F.mse_loss(cap_predictor(noisy), target)
        elif variant == "C_detach":
            target = spherify(z_a).detach()
            noisy, _ = perturb_and_respherify(spherify(z_a), sigma_max=config.sigma_max)
            loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            loss = loss + F.mse_loss(cap_predictor(noisy), target)
        elif variant == "C_ema":
            assert ema_encoder is not None
            with torch.no_grad():
                target = spherify(ema_encoder(view_a))
            noisy, _ = perturb_and_respherify(spherify(z_a), sigma_max=config.sigma_max)
            loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            loss = loss + F.mse_loss(cap_predictor(noisy), target)
        elif variant in {"D0a", "D1a", "D0b", "D1b", "D0c", "D1c"}:
            anchor_kind = {"a": "continuous", "b": "rff", "c": "teacher"}[variant[-1]]
            anchor = get_anchor(anchor_kind, xb, teacher, rff)
            sigma = 0.0 if variant.startswith("D0") else config.sigma_max
            align = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            cap_a = noisy_cap_anchor_loss(z_a, anchor, cap_predictor, sigma_max=sigma)
            cap_b = noisy_cap_anchor_loss(z_b, anchor, cap_predictor, sigma_max=sigma)
            loss = config.lambda_jepa * align + config.lambda_cap * 0.5 * (cap_a + cap_b)
        elif variant == "D3":
            align = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            v_a = spherify(z_a)
            v_b = spherify(z_b)
            v_a, _ = perturb_and_respherify(v_a, sigma_max=config.sigma_max)
            v_b, _ = perturb_and_respherify(v_b, sigma_max=config.sigma_max)
            loss = align + 0.5 * (F.cross_entropy(classifier(v_a), yb) + F.cross_entropy(classifier(v_b), yb))
        elif variant == "F":
            loss = vicreg_loss(z_a, z_b)
        elif variant == "G":
            align = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            loss = align + 0.1 * (sigreg_loss(z_a) + sigreg_loss(z_b))
        else:
            raise ValueError(f"unknown variant: {variant}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if ema_encoder is not None:
            update_ema(ema_encoder, encoder, momentum=config.ema_momentum)
        if step == 0 or (step + 1) % max(1, config.steps // 10) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach())})

    with torch.no_grad():
        z = encoder(x)
        view_a, view_b = make_views(x)
        z_a = encoder(view_a)
        z_b = encoder(view_b)

    metrics = {
        "variant": variant,
        "rankme": rankme(z),
        "uniformity": uniformity_loss(z),
        "cosine": cosine_pair_stats(z),
        "alignment_view_ab": alignment_loss(z_a, z_b),
        "within_class_rankme": within_class_rankme(z, labels),
        "history": history,
        "scatter": [
            {
                "x": float(z[i, 0]),
                "y": float(z[i, 1]),
                "label": int(labels[i]),
            }
            for i in range(min(800, len(z)))
        ],
    }
    return metrics


def write_optional_matplotlib_plot(records: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    variants = [r["variant"] for r in records]
    ranks = [r["rankme"] for r in records]
    mean_cos = [r["cosine"]["mean"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(variants, ranks)
    axes[0].set_title("RankMe")
    axes[0].tick_params(axis="x", rotation=60)
    axes[1].bar(variants, mean_cos)
    axes[1].set_title("Mean off-diagonal cosine")
    axes[1].tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(output_dir / "summary.png", dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Experiment 0 toy sphere-JEPA sanity check")
    parser.add_argument("--preset", choices=["smoke", "cpu", "full"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp0_toy"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="*", default=None, help=f"Subset of variants. Default: {', '.join(VARIANTS)}")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--teacher-steps", type=int, default=None)
    parser.add_argument("--samples-per-cluster", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1, help="CPU torch thread count; 1 keeps smoke tests responsive.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> ToyConfig:
    preset_values = {
        "smoke": {"samples_per_cluster": 12, "steps": 4, "teacher_steps": 4, "batch_size": 32},
        "cpu": {"samples_per_cluster": 128, "steps": 250, "teacher_steps": 250, "batch_size": 256},
        "full": {"samples_per_cluster": 1000, "steps": 1000, "teacher_steps": 1000, "batch_size": 512},
    }[args.preset]
    config = ToyConfig(seed=args.seed, **preset_values)
    if args.steps is not None:
        config.steps = args.steps
    if args.teacher_steps is not None:
        config.teacher_steps = args.teacher_steps
    if args.samples_per_cluster is not None:
        config.samples_per_cluster = args.samples_per_cluster
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    return config


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    unknown = sorted(set(args.variants or VARIANTS) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")

    config = config_from_args(args)
    generator = set_seed(config.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, labels = make_toy_data(config, generator)
    rff = RFFAnchor(2, config.rff_dim, generator=generator)
    teacher = pretrain_teacher(x, config)

    with torch.no_grad():
        anchor_diagnostics = {
            "continuous": anchor_geometry_diagnostics(get_anchor("continuous", x, teacher, rff)),
            "rff": anchor_geometry_diagnostics(get_anchor("rff", x, teacher, rff)),
            "teacher": anchor_geometry_diagnostics(get_anchor("teacher", x, teacher, rff)),
        }

    records = []
    for variant in args.variants or VARIANTS:
        print(f"running {variant}")
        records.append(train_variant(variant, x, labels, config, teacher, rff))

    payload = {
        "config": asdict(config),
        "anchor_geometry": anchor_diagnostics,
        "records": records,
        "headline_contrasts": {
            "noise": "Compare D1a/D1b/D1c against matched D0a/D0b/D0c.",
            "anchor": "Compare D1a/D1b/D1c against C/C_detach/C_ema.",
            "cardinality": "Compare D1 variants against D3 global and within-class RankMe.",
        },
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(payload, f, indent=2)
    write_optional_matplotlib_plot(records, args.output_dir)
    print(f"wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
