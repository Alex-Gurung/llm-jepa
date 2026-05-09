# Experiment 3 - Full LLM-JEPA Fork

This is conditional on Experiment 2 showing that `D_own_1` beats the matched
`D_own_0` sigma=0 control and is competitive with distributional baselines.

The LLM-JEPA fork is wired in `finetune.py` with these additional flags:

```bash
--anchor-type {none,cotrained,ema,frozen}
--sigma-max 0.5
--sphere-radius <float>
--lambda-cap <float>
--cap-mode {own,cross,both}
--ema-momentum 0.99
```

The own-view frozen-anchor variant corresponding to the handoff is:

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

Use `--sigma-max 0.0` for the matched D0 control. Use `--anchor-type
cotrained` for the target-movement control, and `--anchor-type ema
--ema-momentum 0.99` for the EMA control.
