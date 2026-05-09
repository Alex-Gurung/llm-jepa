# Experiment 2 Results

This file is a report template for the frozen-LLM projection-head experiment.

## Required Contrasts

- **D_own_1-vs-D_own_0:** sigma>0 contribution with matched own-view anchors.
- **D_own_1-vs-C:** frozen external anchor contribution over co-trained target movement.
- **D_own_sphere-vs-D_own_raw:** frozen target anisotropy and preprocessing effect.
- **D_own-vs-D_cross:** own-view cap reconstruction vs cross-view cap reconstruction.
- **E1_prefix-vs-E1_no_prefix:** report CE improvement and bits saved, not absolute CE alone.

## Commands

```bash
uv run python experiments/exp2_frozen_llm/cache_hidden_states.py \
  --input datasets/synth_train.jsonl \
  --model-name HuggingFaceTB/SmolLM-135M \
  --pooling last \
  --output results/exp2_frozen_llm/cache.pt

uv run python experiments/exp2_frozen_llm/train_heads.py \
  --cache results/exp2_frozen_llm/cache.pt \
  --anchor-preprocess sphere \
  --variants D_own_0 D_own_1 C F G H

uv run python experiments/exp2_frozen_llm/token_prefix_ce.py \
  --cache results/exp2_frozen_llm/cache.pt \
  --model-name HuggingFaceTB/SmolLM-135M
```
