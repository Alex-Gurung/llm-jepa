from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class CifarAutoencoderTeacher(nn.Module):
    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256 * 4 * 4),
            nn.GELU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.features(x))


def build_dataset(root: Path, train: bool, fake_data: bool):
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor()])
    if fake_data:
        return datasets.FakeData(size=256 if train else 128, image_size=(3, 32, 32), num_classes=10, transform=transform)
    return datasets.CIFAR10(root=str(root), train=train, download=True, transform=transform)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a small unsupervised CIFAR autoencoder teacher.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/exp1_cifar/teacher_autoencoder.pt"))
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(args.data_root, train=True, fake_data=args.fake_data)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = CifarAutoencoderTeacher(feature_dim=args.feature_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for epoch in range(args.epochs):
        total = 0.0
        for images, _ in loader:
            images = images.to(device)
            recon = model(images)
            loss = F.mse_loss(recon, images)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(images)
        print(f"epoch={epoch + 1} recon_mse={total / len(dataset):.5f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "feature_dim": args.feature_dim}, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
