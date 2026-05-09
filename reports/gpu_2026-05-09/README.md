# GPU Run Report - 2026-05-09

This folder contains static HTML reports and summary JSON files from the first
GPU pass. Open the files in `html/` directly in a browser.

## Reports

- `index.html`: aggregate matplotlib/seaborn report for all GPU summaries in
  this folder, including natural variant names, a JEPA-vs-baseline explanation,
  task map, interpretation notes, tables, hypersphere projections, PCA maps,
  cosine histograms, eigenspectra, retrieval heatmaps, and preprocessing
  comparisons.
- `html/exp0_full.html`: full toy run, all variants.
- `html/exp1_cifar_gpu.html`: real CIFAR-10 bridge run on a 10k train / 2k
  test subset.
- `html/exp2_pythia160m_synth_sphere.html`: frozen Pythia-160M on `synth`,
  raw source states with spherified anchors.
- `html/exp2_pythia160m_synth_raw.html`, `norm.html`, `white.html`: D_own
  anchor preprocessing ablations with raw source states.
- `html/exp2_pythia160m_synth_input_white_anchor_white.html`: source-whitened
  and anchor-whitened diagnostic. This is the useful Exp2 run.
- `html/exp3_full_synth.html`: full SmolLM2 synth fine-tune report using
  exact-match downstream evaluation.
- `html/exp3_llama1b_synth.html`: full Llama-3.2-1B-Instruct synth
  fine-tune report using the same downstream evaluation.

Rebuild the aggregate dashboard with:

```bash
uv run --extra experiments python experiments/viewer/build_dashboard.py \
  --report-dir reports/gpu_2026-05-09 \
  --out reports/gpu_2026-05-09/index.html
```

## Exp0 Toy

| Variant | RankMe | Mean Cos | Within-Class RankMe |
|---|---:|---:|---:|
| C | 1.589 | 1.000 | 1.174 |
| C_ema | 6.799 | 0.007 | 2.770 |
| D0a | 5.627 | 0.292 | 3.135 |
| D1a | 6.198 | 0.003 | 3.060 |
| D0b | 6.403 | 0.101 | 2.884 |
| D1b | 6.362 | 0.171 | 2.889 |
| D0c | 6.429 | 0.227 | 3.088 |
| D1c | 7.033 | 0.048 | 3.006 |
| D3 | 7.056 | 0.019 | 3.221 |
| F | 7.538 | 0.036 | 4.920 |
| G | 7.765 | 0.015 | 3.150 |

Read: the co-trained self-anchor (`C`) collapses hard. Sphere noise helps for
continuous and frozen-teacher anchors (`D1a>D0a`, `D1c>D0c`) but not for the
RFF anchor in this run (`D1b` is mixed).

## Exp1 CIFAR Bridge

This is a bridge/sanity run, not the main result.

| Variant | Probe Top-1 | RankMe | Mean Cos | Within-Class RankMe |
|---|---:|---:|---:|---:|
| A | 0.099 | 59.182 | 1.000 | 58.492 |
| B | 0.108 | 2.935 | 0.995 | 4.696 |
| C | 0.098 | 1.637 | 1.000 | 10.683 |
| C_detach | 0.203 | 1.770 | 0.657 | 1.804 |
| C_ema | 0.202 | 4.922 | 0.995 | 17.044 |
| D0 | 0.308 | 11.461 | 0.942 | 16.143 |
| D1 | 0.269 | 12.553 | 0.864 | 14.451 |
| D2 | 0.538 | 19.489 | 0.527 | 20.588 |
| F | 0.370 | 38.074 | 0.648 | 36.358 |
| G | 0.151 | 3.577 | 0.974 | 10.703 |

Read: the noisy frozen anchor (`D1`) improves spread over the clean frozen
anchor (`D0`), but probe accuracy is lower. Since the class-anchor control
(`D2`) uses labels, its high probe score is not evidence for the
sample-specific anchor mechanism.

## Exp2 Frozen LLM

Backbone: `EleutherAI/pythia-160m`, dataset: `synth_train.jsonl`, pooling:
last token.

Raw Pythia hidden states are extremely anisotropic. The raw/sphere/norm/white
anchor-only sweeps mostly collapse because the source states fed into the
projection heads remain anisotropic.

### Source-Whitened Diagnostic

| Variant | Text RankMe | Code RankMe | Text Mean Cos | Direct R@1 | Pred-Code R@1 | Cap R@1 |
|---|---:|---:|---:|---:|---:|---:|
| D_own_0 | 229.024 | 232.593 | 0.058 | 0.000 | 0.397 | 0.986 |
| D_own_1 | 233.613 | 242.855 | 0.036 | 0.002 | 0.458 | 0.979 |
| C | 32.149 | 6.510 | 0.998 | 0.001 | 0.001 | n/a |
| C_ema | 213.163 | 205.584 | 0.040 | 0.001 | 0.319 | n/a |
| F | 145.442 | 146.067 | 0.001 | 0.676 | 0.001 | n/a |
| G | 139.264 | 139.634 | 0.077 | 0.605 | 0.000 | n/a |
| H | 136.486 | 132.462 | 0.016 | 0.001 | 0.238 | n/a |

Read:

- Noisy frozen own-anchor (`D_own_1`) beats clean frozen own-anchor
  (`D_own_0`) on RankMe, mean cosine, and predicted-code retrieval.
- Noisy frozen own-anchor (`D_own_1`) is far stronger than co-trained
  self-anchor (`C`), so external anchoring matters.
- Momentum self-anchor (`C_ema`) is a strong geometry control but trails noisy
  frozen own-anchor on predicted-code retrieval.
- InfoNCE/VICReg still dominate direct embedding retrieval, so the next run
  should compare retrieval spaces carefully rather than claim an accuracy win.
- Input/source whitening appears necessary for this Pythia setup.

## Exp3 Smoke

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, dataset: 256 `synth` examples,
one epoch, no evaluation split.

Completed controls:

- Noisy frozen anchor (`D1`): frozen own-view anchors, `sigma_max=0.5`.
- Clean frozen anchor (`D0`): frozen own-view anchors, `sigma_max=0.0`.
- Co-trained self-anchor (`C`): own-view target produced by the moving model.

This validates the full fine-tuning path only. It is not a downstream result.

Important implementation fixes from this run:

- Transformers was pinned back to 4.55.x because 5.x breaks the original
  `TrainingArguments` usage.
- Cap heads now stay fp32 while the LM can run bf16, avoiding a backward dtype
  error.

## Exp3 Full LLM-JEPA

Task: natural-language-to-regex generation. Metric: strict exact string match
on the 2k held-out `datasets/synth_test.jsonl` examples. The HTML pages also
show first-line exact match, contains-target rate, training traces, and sample
mismatches.

### Llama 1B Full Sweep

Model: `meta-llama/Llama-3.2-1B-Instruct`, dataset:
`datasets/synth_train.jsonl` with 8k examples, four epochs.

| Variant | Exact Match | Contains Target | Matches | Train Loss | Runtime |
|---|---:|---:|---:|---:|---:|
| Regular | 61.45% | 85.95% | 1229/2000 | 0.192 | 834 s |
| C | 49.30% | 85.10% | 986/2000 | 0.209 | 2408 s |
| C_ema | 50.30% | 84.25% | 1006/2000 | 0.691 | 2991 s |
| D0 | 41.10% | 83.75% | 822/2000 | 1.087 | 2906 s |
| D1 | 42.10% | 83.40% | 842/2000 | 1.142 | 2906 s |
| Cross | 47.75% | 82.00% | 955/2000 | 1.501 | 2907 s |
| Both | 40.95% | 83.60% | 819/2000 | 1.272 | 2940 s |

Read:

- The regular supervised fine-tune is the clear downstream winner on strict
  exact match.
- D1 beats D0 by only 1.0 point, so noisy frozen own-view anchoring shows a
  small internal ablation win but not a task win.
- Cross beats D1 by 5.65 points, but still trails the regular fine-tune by
  13.7 points.
- Both is slightly worse than D1, so combining own-view and cross-view caps did
  not help in this run.
- C_ema beats C by 1.0 point, but both self-anchor variants remain below the
  regular fine-tune.

### SmolLM2 Full Sweep

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, same 8k/2k synth split and four
epochs.

All seven variants scored 0/2000 strict exact match. This is a format-capacity
failure for the 135M model on the full regex-generation task, not evidence that
the task metric is broken: the Llama 1B regular fine-tune reaches 61.45% exact
match on the same data and evaluator.

## Recommendation

Do not treat SmolLM2 Exp3 as a task-performance result. The strongest current
downstream result is the Llama 1B full sweep, and it does not show a downstream
accuracy win for the cap/JEPA variants. The useful signal is diagnostic: D1
slightly beats D0, Cross beats D1, and self-anchor EMA slightly beats the
co-trained self-anchor, but the plain supervised fine-tune remains the baseline
to beat in the next GPU pass.
