# GPU Run Report - 2026-05-09

This folder contains static HTML reports and summary JSON files from the first
GPU pass. Open the files in `html/` directly in a browser.

## Reports

- `html/exp0_full.html`: full toy run, all variants.
- `html/exp1_cifar_gpu.html`: real CIFAR-10 bridge run on a 10k train / 2k
  test subset.
- `html/exp2_pythia160m_synth_sphere.html`: frozen Pythia-160M on `synth`,
  raw source states with spherified anchors.
- `html/exp2_pythia160m_synth_raw.html`, `norm.html`, `white.html`: D_own
  anchor preprocessing ablations with raw source states.
- `html/exp2_pythia160m_synth_input_white_anchor_white.html`: source-whitened
  and anchor-whitened diagnostic. This is the useful Exp2 run.

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

Read: co-trained `C` collapses hard. Noise helps for continuous and frozen-AE
anchors (`D1a>D0a`, `D1c>D0c`) but not for RFF in this run (`D1b` is mixed).

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

Read: `D1` improves spread over `D0`, but probe accuracy is lower. Since `D2`
uses labels, its high probe score is not evidence for the sample-specific
anchor mechanism.

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

- `D_own_1 > D_own_0` on RankMe, mean cosine, and predicted-code retrieval.
- `D_own_1 >> C`, so external anchoring matters.
- `C_ema` is a strong geometry control but trails `D_own_1` on predicted-code
  retrieval.
- InfoNCE/VICReg still dominate direct embedding retrieval, so the next run
  should compare retrieval spaces carefully rather than claim an accuracy win.
- Input/source whitening appears necessary for this Pythia setup.

## Exp3 Smoke

Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, dataset: 256 `synth` examples,
one epoch, no evaluation split.

Completed controls:

- `D1`: frozen own-view anchors, `sigma_max=0.5`.
- `D0`: frozen own-view anchors, `sigma_max=0.0`.
- `C`: co-trained own-view target.

This validates the full fine-tuning path only. It is not a downstream result.

Important implementation fixes from this run:

- Transformers was pinned back to 4.55.x because 5.x breaks the original
  `TrainingArguments` usage.
- Cap heads now stay fp32 while the LM can run bf16, avoiding a backward dtype
  error.

## Recommendation

Do not scale Exp3 yet as a task-performance run. The strongest next experiment
is Exp2 on the actual target dataset, with source whitening and explicit
retrieval-space choices. If that reproduces the `D_own_1>D_own_0` and
`D_own_1>C` pattern, then run Exp3 as a geometry-first downstream experiment.
