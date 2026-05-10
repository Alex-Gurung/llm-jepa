# Sphere Cap Ablation Research Plan

This plan starts from the reproduced original LLM-JEPA protocol on NL-RX-SYNTH:

- Base model: `meta-llama/Llama-3.2-1B-Instruct`
- Seeds: `82 23 37 84 4`
- Training: 4 epochs, learning rate `2e-5`, `--predictors 1`, `--last_token -2`
- Effective global batch: 128 via 4 GPUs, per-device batch 4, grad accumulation 8
- Primary metric: strict exact-match regex generation on `datasets/synth_test.jsonl`
- Replication gate: original LLM-JEPA must be compared against regular SFT on the same seeds before interpreting cap-anchor variants

The current replication gate is passed: original LLM-JEPA is materially above regular SFT on the completed five-seed run. The spherical experiments below are therefore not being compared to SFT alone; they must beat or explainably trade off against original LLM-JEPA.

## Core Questions

1. Does adding a spherical cap objective improve the reproduced original LLM-JEPA baseline?
   - Primary contrast: best completed cap variant minus original LLM-JEPA on matched seeds.
   - Success rule: positive paired mean delta and at least 4 of 5 seeds non-negative.

2. Is the observed low-dose gain caused by the spherical noise, or by adding a mild auxiliary cap head?
   - Primary contrast: `c_own_d1_l005 - c_own_d0_l005` on matched seeds.
   - Success rule: noisy low-dose cap must beat clean low-dose cap by paired mean and seed consistency, not merely beat original JEPA.

3. Does spherical noise help independently of anchor/view choice?
   - Primary contrasts: `own_d1 - own_d0`, `cross_d1 - cross_d0`, `both_d1 - both_d0`.
   - Success rule: positive paired mean in at least two cap families, or one family with strong seed-level consistency.

4. Is the useful target own-view reconstruction, cross-view prediction, or both?
   - Primary contrasts: `cross_d1 - own_d1` and `both_d1 - best(own_d1, cross_d1)`.
   - Success rule: prefer the simplest target that improves exact match and seed consistency.

5. Are fixed anchors better than moving anchors?
   - Primary contrasts: `c_own_d1 - own_d1` and `ema_own_d1 - own_d1`.
   - Success rule: moving anchors are useful only if they recover exact match without erasing the cap/noise signal.

6. If cap variants underperform, is the issue cap weight or schedule rather than the spherical idea?
   - Follow-up: run the least-bad cap family with `lambda_cap` in `{0.05, 0.1, 0.25, 0.5}` and a delayed/ramped cap schedule.
   - Success rule: non-negative matched-seed delta versus original LLM-JEPA without being driven by one seed.

## Core Variant Grid

All variants keep original LLM-JEPA active with `--lbd 1.0`; the cap loss is additive through `--lambda-cap 1.0`.

| Variant | Anchor | Cap mode | Sigma | Question |
| --- | --- | --- | --- | --- |
| `own_d0` | frozen | own | 0.0 | Clean own-view control |
| `own_d1` | frozen | own | 0.5 | Own-view spherical noise effect |
| `cross_d0` | frozen | cross | 0.0 | Clean cross-view control |
| `cross_d1` | frozen | cross | 0.5 | Cross-view spherical noise effect |
| `both_d0` | frozen | both | 0.0 | Clean combined-target control |
| `both_d1` | frozen | both | 0.5 | Combined-target spherical noise effect |
| `c_own_d1` | co-trained | own | 0.5 | Moving target control |
| `ema_own_d1` | EMA | own | 0.5 | Momentum target control |

## Follow-Up Dose And Schedule Grid

Run this only after the five-seed `lambda_cap=1.0` grid finishes. The point is to test specific failure modes, not to broaden the search:

1. Did `lambda_cap=1.0` overwhelm the replicated JEPA objective?
   - Script: `experiments/run_sphere_followup_cap_schedule.sh`
   - Variants: `c_own_d1_l005`, `c_own_d1_l010`, `c_own_d1_l025`, `c_own_d1_l050`
   - Contrast: each variant minus original LLM-JEPA and minus `c_own_d1` at `lambda_cap=1.0`.
   - Success rule: a lower dose must improve paired mean exact match and reduce seed-to-seed variance.

2. Is the low-dose gain actually a spherical-noise effect?
   - Variant: `c_own_d0_l005`
   - Contrast: `c_own_d1_l005 - c_own_d0_l005` on matched seeds.
   - Success rule: noisy low-dose cap must beat the clean low-dose cap on paired mean exact match, not merely beat original JEPA.

3. Does delaying the cap objective avoid early language-model adaptation conflict?
   - Variant: `c_own_d1_delay1_ramp3`
   - Contrast: delayed/ramped `c_own_d1` minus constant `c_own_d1`.
   - Success rule: non-negative paired delta on at least 4 of 5 seeds.

4. Is the frozen-anchor failure just too much cap pressure?
   - Variants: `own_d0_l025`, `own_d1_l025`, `cross_d0_l025`, `cross_d1_l025`
   - Contrasts: lower-dose frozen variants minus their `lambda_cap=1.0` counterparts; noisy minus clean at the same lower dose.
   - Success rule: lower dose closes most of the JEPA gap while preserving a clear answer on whether noise helps.

## Sequential GPU Triage

The core success rule requires at least 4 of 5 seeds to be non-negative versus original LLM-JEPA. Once a variant has two negative matched-seed deltas, it can no longer satisfy that rule even if the remaining seeds are positive. At that point the remaining full-dose runs are lower priority than the dose/schedule follow-up.

For efficient GPU use, the first follow-up gate is the failure seed that exposed the largest regression. A lower-dose or delayed-cap setting should first recover that seed toward original LLM-JEPA before expanding to all five seeds.

The first expanded follow-up result is `c_own_d1_l005`: five seeds, mean paired exact-match delta `+1.32 pp` versus original LLM-JEPA, and 4 of 5 non-negative seeds. This passes the exact-match gate.

The matched clean control `c_own_d0_l005` is nearly identical on generation: five seeds, mean paired exact-match delta `+1.28 pp` versus original LLM-JEPA, and 3 of 5 non-negative seeds. The direct noisy-minus-clean contrast is only `+0.04 pp` with 3 of 5 non-negative seeds. Current generation results therefore support a low-dose auxiliary co-trained cap regularizer, but they do not yet support a reliable spherical-noise mechanism.

The dose probe on seed 37 also shows a sharp dose sensitivity: `c_own_d1_l005` reaches `74.00%` exact match, while `c_own_d1_l010`, `c_own_d1_l025`, and `c_own_d1_l050` drop to `69.70%`, `64.25%`, and `64.30%`. The delayed/ramped schedule reaches `71.60%`, better than higher constant doses on that seed but below the constant `0.05` setting. The next dose experiment should therefore compare clean and noisy co-trained caps around the narrow low-dose region before revisiting frozen anchors.

## Current Conclusions

The reproduced original LLM-JEPA baseline remains the control to beat; regular SFT is not the relevant comparator for spherical claims. Full-dose `lambda_cap=1.0` cap variants are not viable under the documented gate because they accumulated enough negative matched-seed deltas to make 4 of 5 non-negative seeds impossible.

The only current downstream improvement is the low-dose co-trained own-view cap. Because the clean control nearly matches the noisy variant, the cleanest interpretation is that a small auxiliary embedding-reconstruction objective may regularize the LM adaptation. The spherical perturbation may still matter for embedding geometry, robustness, or retrieval-like diagnostics, but exact-match generation does not isolate that mechanism.

The next airtight experiment should be a paired low-dose mechanism test:

| Experiment | Variants | Seeds | Decisive contrast |
| --- | --- | --- | --- |
| Low-dose mechanism | `c_own_d0_l005`, `c_own_d1_l005` | all five completed | `d1 - d0`; current result is inconclusive at `+0.04 pp` |
| Dose curve | `c_own_d0_l010`, `c_own_d1_l010`, then `0.025` only if `0.010` survives | all five | Does noise help at the same cap pressure, or do both doses over-constrain? |
| Geometry mechanism | SFT, original JEPA, `c_own_d0_l005`, `c_own_d1_l005` | fixed 512-example subset | RankMe, paired/unpaired cosine gap, retrieval heatmaps, PCA clustering |
| Robustness | same four models | controlled generation perturbations | Does noisy cap improve stability even when clean exact match is similar? |

## Evaluation

Primary claims use exact-match generation only. The HTML report also shows first-line/contains-target diagnostics, but those are debugging aids for malformed generations and should not replace exact match.

The report generated by `experiments/viewer/build_sphere_ablation_report.py` includes:

- Mean exact match with seed standard deviation
- Matched-seed deltas versus original LLM-JEPA
- Clean-versus-noisy paired plots for the frozen cap families
- Seed-by-method exact-match heatmap
- Training-loss traces
- Research questions, decision rules, interpretation, and next experiments embedded in JSON/HTML

Geometry plots are diagnostic rather than decisive. The next geometry pass should extract hidden states for a fixed 512-example test subset from regular SFT, original LLM-JEPA, and the best cap variant, then report PCA maps, cosine histograms, RankMe, paired-vs-unpaired text/code cosine, and retrieval heatmaps.

The first geometry pass now covers seeds 37 and 23 for regular SFT, original LLM-JEPA, `c_own_d0_l005`, and `c_own_d1_l005`. It should be read as a diagnostic, not a win condition. In last-token hidden states, the JEPA and low-dose cap models have much higher text-code cosine than SFT, but paired and unpaired text-code cosine are almost identical. Text-to-code retrieval@1 stays near chance on the 512-example subset. The useful geometry signal so far is representation spread and collapse/anisotropy, not direct nearest-neighbor pairing. A stronger geometry claim needs layer sweeps, pooling sweeps, and possibly embeddings from the predictor/cap spaces rather than only the base causal LM hidden state.

## Resource Expectations

CPU is enough for syntax checks, unit tests, JSON aggregation, and HTML/PNG report generation. It is not enough for faithful full LLM-JEPA training.

The full cap grid is GPU-bound. Frozen-anchor variants are slower than the original replication because each training step also computes reference text/code embeddings. The current runner uses all 4 GPUs for one training job at a time to preserve the reproduced effective batch.

## Interpretation Guardrails

- Do not call a cap variant successful because it beats SFT; original LLM-JEPA is the real baseline.
- Do not overread one seed. Use paired deltas and seed consistency.
- A geometry win without an exact-match win is useful for mechanism debugging, but it is not a downstream win.
- If all `lambda_cap=1.0` cap variants underperform, the clean next test is dose/schedule, not a broad search.
- CIFAR is not part of this LLM-JEPA replication claim. It is only a bridge experiment for cheap representation-learning diagnostics.
