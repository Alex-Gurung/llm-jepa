# Sphere-JEPA Evaluation Guide

This guide is the source of truth for what each experiment is expected to
answer, which variants must be compared, and how to decide whether a run is
scientifically useful. The scripts write JSON summaries that can be rendered
with `experiments/viewer/build_viewer.py`; the JSON is the canonical result
artifact and the HTML is for inspection.

## Central Questions

The project tests whether sphere normalization plus noise is useful because it
forces recovery of a sample-specific external anchor, and whether the noise
itself contributes beyond clean anchor reconstruction.

Every report should explicitly answer two headline contrasts:

1. **Anchor contribution:** compare the best noisy external-anchor variant
   against co-trained targets. In the handoff notation this is D1-vs-C.
2. **Noise contribution:** compare sigma>0 against sigma=0 with the same
   external anchor. In the handoff notation this is D1-vs-D0.

Do not treat high absolute RankMe or low cross-entropy as sufficient. The
matched controls are the evidence.

## Shared Metrics

Use the same metric names across experiments where possible.

- `rankme`: effective representation rank. Higher usually means less collapse,
  but low-dimensional data may not need the full embedding dimension.
- `within_class_rankme`: required whenever labels are available. This catches
  class-level spread with within-class collapse.
- `uniformity`: Wang-Isola uniformity on normalized features. More negative is
  generally more uniform on the sphere.
- `cosine.mean`, `cosine.std`, `cosine.p95`: off-diagonal pairwise cosine
  statistics. Collapse usually shows up as high mean cosine and high p95.
- `alignment_view_ab`: normalized squared distance between paired views. Lower
  is better only if the representation is not collapsed.
- `anchor_geometry`: diagnostics on frozen targets before training. Always
  report RankMe, top eigenvalue mass ratio, mean norm, uniformity after
  normalization, and cosine stats.
- Retrieval metrics in Experiment 2: use Recall@1 and Recall@10 on held-out
  paired data. For asymmetric cap-to-anchor variants, retrieve in the frozen
  anchor space.
- Token-prefix metrics in Experiment 2 E1: report CE improvement and bits saved
  over `E1_no_prefix`, not absolute CE alone.

## Result Artifacts

Each real run should leave these files in `results/<experiment_name>/`:

- `summary*.json`: canonical machine-readable metrics and run config.
- `index.html`: optional self-contained viewer created from the summary JSON.
- `RESULTS.md`: short human report that states the required contrasts and
  interprets them.
- Checkpoints only when needed to reproduce later evaluations. Large
  checkpoints should not be committed.

Viewer command:

```bash
uv run --no-sync python experiments/viewer/build_viewer.py \
  results/<experiment>/summary.json \
  --out results/<experiment>/index.html
```

The viewer is intentionally static and browser-only. For summaries that include
enhanced diagnostics, inspect these panels before looking at downstream task
metrics:

- **PCA scatter:** checks whether variants visibly collapse, spread by class,
  or form one diffuse cloud.
- **Cosine histograms:** checks the full pairwise similarity distribution, not
  just mean/p95. Collapse appears as a distribution concentrated near 1.
- **Eigenspectra:** checks whether representation mass is concentrated in a few
  dimensions. A dominant first eigenvalue is an anisotropy warning.
- **Retrieval heatmaps:** checks whether paired examples light up on the
  diagonal. Off-diagonal blocks indicate clustering or shortcuts.
- **Anchor diagnostics:** checks whether the frozen target space is already
  anisotropic before training starts.

## Experiment 0 - Toy 2D Sanity Check

### Purpose

This is the fast mechanism check. It should reveal whether the implementation
can distinguish collapsed co-trained targets, sample-specific external anchors,
sigma=0 anchored reconstruction, and low-cardinality class anchors.

### Required Variants

- `A`: plain JEPA, no sphere.
- `B`: sphere-only JEPA.
- `C`: sphere plus co-trained target, no detach.
- `C_detach`: detached co-trained target.
- `C_ema`: EMA co-trained target, momentum 0.99.
- `D0a`: sigma=0, reconstruct continuous own-view point `x`.
- `D1a`: sigma>0, reconstruct continuous own-view point `x`.
- `D0b`: sigma=0, reconstruct random Fourier features.
- `D1b`: sigma>0, reconstruct random Fourier features.
- `D0c`: sigma=0, reconstruct frozen autoencoder teacher features.
- `D1c`: sigma>0, reconstruct frozen autoencoder teacher features.
- `D3`: sigma>0, predict cluster ID as low-cardinality diagnostic.
- `F`: VICReg baseline.
- `G`: SIGReg baseline.

### Evaluation

Run the full toy set first, then inspect the viewer.

```bash
uv run --no-sync python experiments/exp0_toy/run.py \
  --preset cpu \
  --output-dir results/exp0_toy
uv run --no-sync python experiments/viewer/build_viewer.py \
  results/exp0_toy/summary.json \
  --out results/exp0_toy/index.html
```

The report must state:

- Whether D1a/b/c have healthier RankMe and cosine stats than A/B/C/D3.
- Whether each D1 variant beats its matched D0 variant. This is the noise
  mechanism claim.
- Whether D1 variants beat C/C_detach/C_ema. This is the external-anchor claim.
- Whether D3 has misleading global RankMe but low within-class RankMe.
- Whether F/G behave like reasonable non-collapsed distributional baselines.

### Interpretation

- **D1 > D0 and D1 > C:** proceed to larger settings.
- **D1 ~= D0 but D1 > C:** anchoring works, but noise is not adding much.
- **D1 <= C or D1 collapses:** check `spherify`, sigma handling, anchor
  freezing, and cap predictor dimensions before scaling.
- **D3 looks strong globally:** use within-class RankMe as the deciding
  diagnostic.

## Experiment 1 - CIFAR-10 Small JEPA

### Purpose

This checks whether the toy mechanism transfers to image augmentations and
learned visual representations. Scientific D0/D1 runs require an unsupervised
teacher, not a classifier teacher.

### Required Variants

- `A`, `B`, `C`, `C_detach`, `C_ema`: same controls as Experiment 0.
- `D0`: sigma=0, reconstruct frozen unsupervised teacher features of the own
  view.
- `D1`: sigma>0, reconstruct frozen unsupervised teacher features of the own
  view.
- `D2`: sigma>0, predict CIFAR-10 class label as the low-cardinality diagnostic.
- `F`: VICReg.
- `G`: SIGReg.

### Teacher Requirement

Use `experiments/exp1_cifar/train_teacher_autoencoder.py` for a small
unsupervised autoencoder teacher, or replace it with a stronger SimCLR teacher.
Do not use classifier penultimate features as the main D0/D1 anchor, because a
classifier can discard within-class information.

Smoke test only:

```bash
uv run --no-sync python experiments/exp1_cifar/train_teacher_autoencoder.py \
  --fake-data \
  --epochs 1 \
  --output results/exp1_cifar/teacher_smoke.pt
uv run --no-sync python experiments/exp1_cifar/run.py \
  --fake-data \
  --cpu \
  --arch tiny \
  --epochs 1 \
  --probe-epochs 1 \
  --batch-size 64 \
  --teacher-ckpt results/exp1_cifar/teacher_smoke.pt \
  --variants D0 D1
```

Real run:

```bash
uv run --no-sync python experiments/exp1_cifar/train_teacher_autoencoder.py \
  --epochs 20 \
  --output results/exp1_cifar/teacher_autoencoder.pt
uv run --no-sync python experiments/exp1_cifar/run.py \
  --teacher-ckpt results/exp1_cifar/teacher_autoencoder.pt \
  --variants A B C C_detach C_ema D0 D1 D2 F G \
  --output-dir results/exp1_cifar
```

### Evaluation

The report must state:

- Linear probe top-1 for each variant.
- D1-vs-D0 on RankMe, uniformity, cosine stats, and probe accuracy.
- D1-vs-C/C_ema on the same metrics.
- D1-vs-D2, including D2 within-class RankMe.
- Anchor geometry diagnostics for the frozen teacher feature distribution.

### Interpretation

- **D1 > D0 and D1 matches F/G:** mechanism transfers to images.
- **D1 ~= D0:** anchoring transfers, noise contribution is weak.
- **D1 ~= C_ema:** EMA target may be doing most of the work.
- **D2 has low within-class RankMe:** confirms sample-specific cardinality
  matters beyond class labels.

## Experiment 2 - Frozen LLM Projection Heads

### Purpose

This is the primary language experiment before full fine-tuning. It isolates
the representation geometry and retrieval behavior using cached frozen LLM
hidden states and lightweight projection/cap heads.

### Required Setup

- Backbone: frozen SmolLM-135M or Pythia-160M.
- Pooling: last token of last layer by default. Mean pooling is an ablation.
- Cache both NL/text and code hidden states once with
  `cache_hidden_states.py`.
- Train `g_text`, `g_code`, predictors, and cap predictors on cached tensors.
- Run anchor preprocessing modes: `raw`, `norm`, `white`, `sphere`, at least
  for the D_own family.

### Required Variants

- `A`: plain JEPA.
- `B`: sphere-only JEPA.
- `C`, `C_detach`, `C_ema`: co-trained target controls.
- `D_own_0`: sigma=0 own-view cap reconstruction plus clean cosine alignment.
- `D_own_1`: sigma>0 own-view cap reconstruction plus clean cosine alignment.
- `D_cross_0`, `D_cross_1`: text cap predicts paired code anchor.
- `D_cross_sym_0`, `D_cross_sym_1`: symmetric cross-view cap reconstruction.
- `F`: CLIP-style InfoNCE.
- `G`: VICReg.
- `H`: SIGReg.

### Commands

Cache:

```bash
uv run --no-sync python experiments/exp2_frozen_llm/cache_hidden_states.py \
  --input datasets/synth_train.jsonl \
  --model-name HuggingFaceTB/SmolLM-135M \
  --pooling last \
  --output results/exp2_frozen_llm/cache.pt
```

Train heads for one preprocessing mode:

```bash
uv run --no-sync python experiments/exp2_frozen_llm/train_heads.py \
  --cache results/exp2_frozen_llm/cache.pt \
  --anchor-preprocess sphere \
  --variants D_own_0 D_own_1 C C_detach C_ema D_cross_0 D_cross_1 F G H \
  --output-dir results/exp2_frozen_llm
```

Repeat with `--anchor-preprocess raw`, `norm`, and `white`.

### Evaluation

The report must state:

- Anchor geometry for raw text/code hidden states before training.
- Anchor geometry after each preprocessing mode.
- D_own_1-vs-D_own_0 on text/code RankMe, uniformity, cosine stats, and
  retrieval.
- D_own_1-vs-C/C_ema on the same metrics.
- D_own_sphere-vs-D_own_raw to decide whether anisotropy preprocessing helps.
- D_own-vs-D_cross to decide whether own-view reconstruction is sufficient.
- Retrieval@1 and Retrieval@10 on held-out text-to-code pairs.

### Token Anchor E1

The E1 no-prefix baseline is required before interpreting prefix CE.

```bash
uv run --no-sync python experiments/exp2_frozen_llm/token_prefix_ce.py \
  --cache results/exp2_frozen_llm/cache.pt \
  --model-name HuggingFaceTB/SmolLM-135M \
  --output results/exp2_frozen_llm/e1_prefix.json
```

Report:

- `E1_no_prefix_ce`
- `E1_prefix_ce`
- `ce_improvement`
- `bits_saved`

If CE improvement is near zero, the prefix is not carrying useful information
even if absolute CE looks acceptable.

### Interpretation

- **D_own_1 > D_own_0 and D_own_1 > C:** green-light Experiment 3.
- **D_own_1 ~= D_own_0:** language setting supports anchored reconstruction,
  not necessarily the noise mechanism.
- **D_own_sphere >> D_own_raw:** target anisotropy matters. Use sphere or white
  preprocessing downstream.
- **D_own ~= D_cross:** prefer the simpler own-view design.
- **E1_prefix ~= E1_no_prefix:** token-anchor prefix is not informative enough.

## Experiment 3 - Full LLM-JEPA Fork

### Purpose

This is conditional. Only run full fine-tuning after Experiment 2 shows a real
reason to pay the compute cost.

### Required Setup

Use the original LLM-JEPA fine-tuning path with the added cap-anchor flags in
`finetune.py`.

Primary D1 command shape:

```bash
torchrun --nproc_per_node=8 finetune.py \
  --train_file datasets/synth_train.jsonl \
  --output_dir ./fine-tuned \
  --model_name meta-llama/Llama-3.2-1B-Instruct \
  --last_token -2 \
  --lbd 1.0 \
  --lambda-cap 1.0 \
  --anchor-type frozen \
  --cap-mode own \
  --sigma-max 0.5 \
  --predictors 0
```

Matched controls:

- D0: same command with `--sigma-max 0.0`.
- C: `--anchor-type cotrained`.
- C_ema: `--anchor-type ema --ema-momentum 0.99`.
- Cross-view ablation: `--cap-mode cross`.
- Both directions: `--cap-mode both`.

### Evaluation

The report must state:

- Downstream task metric from `evaluate.py`.
- RankMe, uniformity, and cosine stats if representation dumps are available.
- D1-vs-D0 noise contribution.
- D1-vs-C/C_ema anchor contribution.
- Whether the intervention improves downstream metrics, improves geometry only,
  or fails both.

### Interpretation

- **Downstream improvement:** proceed to seed sweeps and hyperparameter sweeps.
- **Geometry-only improvement:** still useful, but document as a geometry
  result rather than a task win.
- **No downstream or geometry improvement:** do not scale without revisiting
  Experiment 2 and anchor preprocessing.

## Reporting Checklist

Every `RESULTS.md` should include:

- Exact command line and commit SHA.
- Dataset, split, seed, and number of examples.
- Variant table with all required metrics.
- Anchor geometry table for any external target.
- D1-vs-D0 paragraph.
- D1-vs-C paragraph.
- Sample-specific vs low-cardinality paragraph where labels exist.
- E1 CE improvement over no-prefix where token anchors are used.
- Known limitations, including whether the run was smoke, CPU-only, fake-data,
  or GPU-scale.
