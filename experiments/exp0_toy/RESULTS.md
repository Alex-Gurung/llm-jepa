# Experiment 0 Results

This file is intentionally a report template until a full CPU run has been
completed.

## Required Contrasts

- **D1-vs-D0 noise contribution:** compare D1a/D1b/D1c to matched D0a/D0b/D0c.
- **D1-vs-C anchor contribution:** compare D1a/D1b/D1c to C, C_detach, and C_ema.
- **Sample-specific vs class-level:** compare D1 variants to D3 global RankMe and D3 within-class RankMe.

## Required Diagnostics

- Anchor geometry for continuous, random Fourier, and frozen autoencoder anchors.
- RankMe, uniformity, cosine pair stats, and within-class RankMe for each variant.

## CPU Commands

```bash
uv run python experiments/exp0_toy/run.py --preset smoke --output-dir results/exp0_smoke
uv run python experiments/viewer/build_viewer.py results/exp0_smoke/summary.json --out results/exp0_smoke/index.html
```
