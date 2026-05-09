# Sphere-JEPA Experiments

The experiment code is designed so small CPU smoke tests work now and the same
entry points scale to GPU runs later.

Read [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) before interpreting any result.
It defines the required controls, metrics, headline contrasts, report
checklists, and go/no-go decisions for each experiment.

## Experiment 0

```bash
uv run --no-sync python experiments/exp0_toy/run.py --preset smoke --output-dir results/exp0_smoke
uv run --no-sync python experiments/viewer/build_viewer.py results/exp0_smoke/summary.json --out results/exp0_smoke/index.html
```

For a stronger CPU mechanics run with all toy variants:

```bash
uv run --no-sync python experiments/exp0_toy/run.py \
  --preset cpu \
  --steps 150 \
  --teacher-steps 150 \
  --samples-per-cluster 192 \
  --output-dir results/exp0_cpu_all
uv run --no-sync python experiments/viewer/build_viewer.py \
  results/exp0_cpu_all/summary.json \
  --out results/exp0_cpu_all/index.html
```

## Experiment 1

The CIFAR scripts support a fast `--fake-data` smoke path and a real CIFAR-10
path for GPU runs. Train or provide an unsupervised teacher checkpoint before
using D0/D1 as scientific results.

CPU fake-data mechanics run:

```bash
uv run --no-sync python experiments/exp1_cifar/train_teacher_autoencoder.py \
  --fake-data \
  --epochs 2 \
  --batch-size 64 \
  --output results/exp1_cifar_cpu/teacher_fake.pt
uv run --no-sync python experiments/exp1_cifar/run.py \
  --fake-data \
  --cpu \
  --arch tiny \
  --epochs 2 \
  --probe-epochs 2 \
  --batch-size 64 \
  --teacher-ckpt results/exp1_cifar_cpu/teacher_fake.pt \
  --variants A B C C_detach C_ema D0 D1 D2 F G \
  --output-dir results/exp1_cifar_cpu
uv run --no-sync python experiments/viewer/build_viewer.py \
  results/exp1_cifar_cpu/summary.json \
  --out results/exp1_cifar_cpu/index.html
```

Small real-CIFAR CPU subset:

```bash
uv run --no-sync python experiments/exp1_cifar/train_teacher_autoencoder.py \
  --train-limit 1024 \
  --epochs 1 \
  --batch-size 64 \
  --output results/exp1_cifar_real_cpu/teacher_subset.pt
uv run --no-sync python experiments/exp1_cifar/run.py \
  --cpu \
  --arch tiny \
  --train-limit 1024 \
  --test-limit 512 \
  --epochs 1 \
  --probe-epochs 1 \
  --batch-size 64 \
  --teacher-ckpt results/exp1_cifar_real_cpu/teacher_subset.pt \
  --variants D0 D1 \
  --output-dir results/exp1_cifar_real_cpu
```

## Experiment 2

Cache frozen LLM hidden states once, then train lightweight heads over the
cached tensors. The primary pooling mode is `last`.

For CPU-only viewer validation, a synthetic cache can exercise the head
training, retrieval tables, heatmaps, and preprocessing ablations without
downloading an LLM.

## Experiment 3

`finetune.py` contains the LLM-JEPA cap-anchor flags. This should only be run
after Experiment 2 justifies moving into full fine-tuning.

## Viewer

`experiments/viewer/build_viewer.py` renders a static HTML report from a summary
JSON. When the summary contains the enhanced diagnostics, the viewer shows:

- headline D1-vs-D0 and D1-vs-C contrast tables
- metric bar charts
- anchor geometry tables
- anchor and embedding eigenspectra
- cosine similarity histograms
- PCA scatter plots for learned embeddings
- text/code PCA scatter for frozen-LLM heads
- retrieval tables
- small cosine-similarity heatmaps for paired retrieval and cap-to-anchor spaces
