# Experiment 1 Results

This file is a report template until GPU-scale CIFAR-10 runs are complete.

## Required Contrasts

- **D1-vs-D0 noise contribution:** same frozen unsupervised teacher anchor, sigma>0 vs sigma=0.
- **D1-vs-C anchor contribution:** frozen external teacher anchor vs co-trained targets.
- **Sample-specific vs class-level:** D1 vs D2, including within-class RankMe for D2.

## Required Diagnostics

- Linear probe top-1 on CIFAR-10 test.
- RankMe, uniformity, cosine pair stats.
- Within-class RankMe wherever labels are used.
- Anchor geometry diagnostics for the frozen teacher features before D0/D1 training.

## CPU Smoke Commands

```bash
uv run python experiments/exp1_cifar/train_teacher_autoencoder.py --fake-data --epochs 1 --output results/exp1_cifar/teacher_smoke.pt
uv run python experiments/exp1_cifar/run.py --fake-data --cpu --arch tiny --epochs 1 --probe-epochs 1 --batch-size 64 --teacher-ckpt results/exp1_cifar/teacher_smoke.pt --variants D0 D1
uv run python experiments/viewer/build_viewer.py results/exp1_cifar/summary.json --out results/exp1_cifar/index.html
```
