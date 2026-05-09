# Sphere-JEPA Experiments

The experiment code is designed so small CPU smoke tests work now and the same
entry points scale to GPU runs later.

## Experiment 0

```bash
uv run python experiments/exp0_toy/run.py --preset smoke --output-dir results/exp0_smoke
uv run python experiments/viewer/build_viewer.py results/exp0_smoke/summary.json --out results/exp0_smoke/index.html
```

## Experiment 1

The CIFAR scripts support a fast `--fake-data` smoke path and a real CIFAR-10
path for GPU runs. Train or provide an unsupervised teacher checkpoint before
using D0/D1 as scientific results.

## Experiment 2

Cache frozen LLM hidden states once, then train lightweight heads over the
cached tensors. The primary pooling mode is `last`.

## Experiment 3

`finetune.py` contains the LLM-JEPA cap-anchor flags. This should only be run
after Experiment 2 justifies moving into full fine-tuning.
