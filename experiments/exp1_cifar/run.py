from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from train_teacher_autoencoder import CifarAutoencoderTeacher
from sphere_jepa.heads import MLP, make_ema_copy, update_ema
from sphere_jepa.losses import clean_jepa_alignment, noisy_cap_anchor_loss, sigreg_loss, vicreg_loss
from sphere_jepa.metrics import (
    anchor_geometry_diagnostics,
    cosine_pair_stats,
    embedding_visual_diagnostics,
    rankme,
    uniformity_loss,
    within_class_rankme,
)
from sphere_jepa.spherify import perturb_and_respherify, spherify


VARIANTS = ["A", "B", "C", "C_detach", "C_ema", "D0", "D1", "D2", "F", "G"]


class TwoViewDataset(Dataset):
    def __init__(self, base: Dataset, transform) -> None:
        self.base = base
        self.transform = transform

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        image, label = self.base[idx]
        return self.transform(image), self.transform(image), int(label)


class CifarEncoder(nn.Module):
    def __init__(self, emb_dim: int = 128, arch: str = "resnet18") -> None:
        super().__init__()
        if arch == "tiny":
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, emb_dim),
            )
        elif arch == "resnet18":
            from torchvision.models import resnet18

            model = resnet18(weights=None)
            model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            model.maxpool = nn.Identity()
            model.fc = nn.Linear(model.fc.in_features, emb_dim)
            self.net = model
        else:
            raise ValueError(f"unknown arch: {arch}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ssl_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=(0.45, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ]
    )


def eval_transform():
    from torchvision import transforms

    return transforms.Compose([transforms.ToTensor()])


def build_base_dataset(root: Path, train: bool, fake_data: bool):
    from torchvision import datasets

    if fake_data:
        size = 256 if train else 128
        return datasets.FakeData(size=size, image_size=(3, 32, 32), num_classes=10, transform=None)
    return datasets.CIFAR10(root=str(root), train=train, download=True, transform=None)


def maybe_limit_dataset(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, list(range(limit)))


def load_teacher(path: Path | None, feature_dim: int, allow_random: bool, device: torch.device) -> CifarAutoencoderTeacher:
    teacher = CifarAutoencoderTeacher(feature_dim=feature_dim)
    if path is not None and path.exists():
        payload = torch.load(path, map_location="cpu")
        teacher.load_state_dict(payload["state_dict"])
    elif not allow_random:
        raise FileNotFoundError("D0/D1 require an unsupervised teacher checkpoint; pass --teacher-ckpt or --allow-random-teacher for smoke tests")
    else:
        print("warning: using a random frozen teacher; this is only valid for smoke tests")
    teacher.to(device).eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


@torch.no_grad()
def collect_embeddings(model: nn.Module, dataset: Dataset, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    from torchvision import transforms

    class EvalDataset(Dataset):
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, idx: int):
            image, label = dataset[idx]
            return transforms.ToTensor()(image), int(label)

    loader = DataLoader(EvalDataset(), batch_size=batch_size, shuffle=False, num_workers=0)
    zs, ys = [], []
    model.eval()
    for images, labels in loader:
        zs.append(model(images.to(device)).cpu())
        ys.append(labels)
    model.train()
    return torch.cat(zs), torch.cat(ys)


def train_linear_probe(
    encoder: nn.Module,
    train_base: Dataset,
    test_base: Dataset,
    device: torch.device,
    *,
    batch_size: int,
    epochs: int,
) -> float:
    train_z, train_y = collect_embeddings(encoder, train_base, device, batch_size)
    test_z, test_y = collect_embeddings(encoder, test_base, device, batch_size)
    probe = nn.Linear(train_z.shape[-1], 10)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-2)
    for _ in range(epochs):
        perm = torch.randperm(len(train_z))
        for start in range(0, len(train_z), batch_size):
            idx = perm[start:start + batch_size]
            loss = F.cross_entropy(probe(train_z[idx]), train_y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = probe(test_z).argmax(dim=-1)
    return (pred == test_y).float().mean().item()


@torch.no_grad()
def teacher_anchor_diagnostics(args: argparse.Namespace, base: Dataset) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    teacher = load_teacher(args.teacher_ckpt, args.teacher_dim, args.allow_random_teacher or args.fake_data, device)
    from torchvision import transforms

    class EvalDataset(Dataset):
        def __len__(self) -> int:
            return min(len(base), args.diagnostic_samples)

        def __getitem__(self, idx: int):
            image, _ = base[idx]
            return transforms.ToTensor()(image)

    loader = DataLoader(EvalDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0)
    feats = []
    for images in loader:
        feats.append(teacher.features(images.to(device)).cpu())
    return {"teacher_features": anchor_geometry_diagnostics(torch.cat(feats))}


def train_variant(args: argparse.Namespace, variant: str, train_loader: DataLoader, train_base: Dataset, test_base: Dataset) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    encoder = CifarEncoder(args.emb_dim, arch=args.arch).to(device)
    predictor = MLP(args.emb_dim, args.emb_dim, hidden_dim=args.emb_dim * 2).to(device)
    cotrain_cap = MLP(args.emb_dim, args.emb_dim, hidden_dim=args.emb_dim * 2).to(device)
    anchor_cap = MLP(args.emb_dim, args.teacher_dim, hidden_dim=args.emb_dim * 2).to(device)
    classifier = MLP(args.emb_dim, 10, hidden_dim=args.emb_dim * 2).to(device)
    ema_encoder = make_ema_copy(encoder).to(device) if variant == "C_ema" else None
    teacher = load_teacher(args.teacher_ckpt, args.teacher_dim, args.allow_random_teacher or args.fake_data, device)

    params = list(encoder.parameters())
    if variant != "F":
        params.extend(predictor.parameters())
    if variant in {"C", "C_detach", "C_ema"}:
        params.extend(cotrain_cap.parameters())
    if variant in {"D0", "D1"}:
        params.extend(anchor_cap.parameters())
    if variant == "D2":
        params.extend(classifier.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        for view_a, view_b, labels in train_loader:
            view_a = view_a.to(device)
            view_b = view_b.to(device)
            labels = labels.to(device)
            z_a = encoder(view_a)
            z_b = encoder(view_b)
            if variant == "A":
                loss = F.mse_loss(predictor(z_a), z_b.detach())
            elif variant == "B":
                loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
            elif variant == "C":
                target = spherify(z_a)
                noisy, _ = perturb_and_respherify(target, sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine") + F.mse_loss(cotrain_cap(noisy), target)
            elif variant == "C_detach":
                target = spherify(z_a).detach()
                noisy, _ = perturb_and_respherify(spherify(z_a), sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine") + F.mse_loss(cotrain_cap(noisy), target)
            elif variant == "C_ema":
                assert ema_encoder is not None
                with torch.no_grad():
                    target = spherify(ema_encoder(view_a))
                noisy, _ = perturb_and_respherify(spherify(z_a), sigma_max=args.sigma_max)
                loss = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine") + F.mse_loss(cotrain_cap(noisy), target)
            elif variant in {"D0", "D1"}:
                with torch.no_grad():
                    anchor_a = teacher.features(view_a)
                    anchor_b = teacher.features(view_b)
                sigma = 0.0 if variant == "D0" else args.sigma_max
                align = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
                cap_loss = noisy_cap_anchor_loss(z_a, anchor_a, anchor_cap, sigma_max=sigma)
                cap_loss = cap_loss + noisy_cap_anchor_loss(z_b, anchor_b, anchor_cap, sigma_max=sigma)
                loss = args.lambda_jepa * align + args.lambda_cap * 0.5 * cap_loss
            elif variant == "D2":
                align = clean_jepa_alignment(z_a, z_b, predictor, metric="cosine")
                loss = align + 0.5 * (F.cross_entropy(classifier(spherify(z_a)), labels) + F.cross_entropy(classifier(spherify(z_b)), labels))
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
                update_ema(ema_encoder, encoder, args.ema_momentum)
        print(f"variant={variant} epoch={epoch + 1} loss={float(loss.detach()):.4f}")

    z, y = collect_embeddings(encoder, train_base, device, args.batch_size)
    probe_acc = train_linear_probe(encoder, train_base, test_base, device, batch_size=args.batch_size, epochs=args.probe_epochs)
    visuals = embedding_visual_diagnostics(z, y, max_points=args.visual_samples)
    return {
        "variant": variant,
        "linear_probe_top1": probe_acc,
        "rankme": rankme(z),
        "uniformity": uniformity_loss(z),
        "cosine": cosine_pair_stats(z),
        "within_class_rankme": within_class_rankme(z, y),
        "scatter": visuals["pca_scatter"],
        **visuals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 CIFAR-10 small JEPA")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp1_cifar"))
    parser.add_argument("--teacher-ckpt", type=Path, default=None)
    parser.add_argument("--teacher-dim", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--arch", choices=["resnet18", "tiny"], default="resnet18")
    parser.add_argument("--variants", nargs="*", default=VARIANTS)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--lambda-jepa", type=float, default=1.0)
    parser.add_argument("--lambda-cap", type=float, default=1.0)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional subset size for CPU/debug runs.")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional test subset size for CPU/debug runs.")
    parser.add_argument("--allow-random-teacher", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--diagnostic-samples", type=int, default=2048)
    parser.add_argument("--visual-samples", type=int, default=800)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    unknown = sorted(set(args.variants) - set(VARIANTS))
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_base = maybe_limit_dataset(build_base_dataset(args.data_root, train=True, fake_data=args.fake_data), args.train_limit)
    test_base = maybe_limit_dataset(build_base_dataset(args.data_root, train=False, fake_data=args.fake_data), args.test_limit)
    train_ds = TwoViewDataset(train_base, ssl_transform())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    anchor_geometry = teacher_anchor_diagnostics(args, train_base) if any(v in {"D0", "D1"} for v in args.variants) else {}

    records = []
    for variant in args.variants:
        records.append(train_variant(args, variant, train_loader, train_base, test_base))

    payload = {
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "anchor_geometry": anchor_geometry,
        "records": records,
        "headline_contrasts": {
            "noise": "D1 vs D0 with the same frozen teacher anchor.",
            "anchor": "D1 vs C/C_detach/C_ema.",
            "cardinality": "D1 vs D2 global and within-class RankMe.",
        },
    }
    out = args.output_dir / "summary.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
