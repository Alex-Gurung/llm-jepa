#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-results/replicate_llm_jepa_sphere_ablation}
MODEL=${MODEL:-meta-llama/Llama-3.2-1B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-datasets/synth_train.jsonl}
TEST_FILE=${TEST_FILE:-datasets/synth_test.jsonl}
SEEDS=${SEEDS:-"82 23 37 84 4"}
VARIANTS=${VARIANTS:-"c_own_d0_l005 c_own_d1_l005 c_own_d1_l010 c_own_d1_l025 c_own_d1_l050 c_own_d1_delay1_ramp3 own_d0_l025 own_d1_l025 cross_d0_l025 cross_d1_l025"}

mkdir -p "$ROOT/logs" "$ROOT/eval"

variant_args() {
  local variant=$1
  case "$variant" in
    c_own_d0_l005)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.0 --lambda-cap 0.05"
      ;;
    c_own_d1_l005)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.5 --lambda-cap 0.05"
      ;;
    c_own_d1_l010)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.5 --lambda-cap 0.10"
      ;;
    c_own_d1_l025)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.5 --lambda-cap 0.25"
      ;;
    c_own_d1_l050)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.5 --lambda-cap 0.50"
      ;;
    c_own_d1_delay1_ramp3)
      echo "--anchor-type cotrained --cap-mode own --sigma-max 0.5 --lambda-cap 1.0 --cap-delay-epochs 1.0 --cap-warmup-epochs 3.0"
      ;;
    own_d0_l025)
      echo "--anchor-type frozen --cap-mode own --sigma-max 0.0 --lambda-cap 0.25"
      ;;
    own_d1_l025)
      echo "--anchor-type frozen --cap-mode own --sigma-max 0.5 --lambda-cap 0.25"
      ;;
    cross_d0_l025)
      echo "--anchor-type frozen --cap-mode cross --sigma-max 0.0 --lambda-cap 0.25"
      ;;
    cross_d1_l025)
      echo "--anchor-type frozen --cap-mode cross --sigma-max 0.5 --lambda-cap 0.25"
      ;;
    *)
      echo "unknown variant: $variant" >&2
      return 1
      ;;
  esac
}

line_count() {
  local file=$1
  if [[ -f "$file" ]]; then
    wc -l < "$file"
  else
    echo 0
  fi
}

run_variant() {
  local seed=$1
  local variant=$2
  local model_dir="$ROOT/seed${seed}_${variant}"
  local eval_file="$ROOT/eval/seed${seed}_${variant}.jsonl"
  local train_log="$ROOT/logs/seed${seed}_${variant}_train.log"
  local eval_log="$ROOT/logs/seed${seed}_${variant}_eval.log"
  local args
  args=$(variant_args "$variant")

  if [[ ! -f "$model_dir/trainer_state.json" ]]; then
    echo "=== TRAIN seed=$seed variant=$variant ==="
    # shellcheck disable=SC2086
    uv run --no-sync torchrun --nproc_per_node=4 finetune.py \
      --train_file "$TRAIN_FILE" \
      --model_name "$MODEL" \
      --output_dir "$model_dir" \
      --batch_size 4 \
      --grad_accum 8 \
      --learning_rate 2e-5 \
      --num_epochs 4 \
      --eval_steps 10 \
      --last_token -2 \
      --lbd 1.0 \
      --predictors 1 \
      --save_strategy no \
      --finetune_seed "$seed" \
      $args 2>&1 | tee "$train_log"
  else
    echo "=== SKIP TRAIN seed=$seed variant=$variant ==="
  fi

  if [[ ! -f "$eval_file" || "$(line_count "$eval_file")" -lt 2000 ]]; then
    echo "=== EVAL seed=$seed variant=$variant ==="
    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python evaluate.py \
      --model_name "$model_dir" \
      --original_model_name "$MODEL" \
      --input_file "$TEST_FILE" \
      --output_file "$eval_file" \
      --nosplit_data \
      --max_new_tokens 64 \
      --generation_batch_size 32 \
      --split_tune_untune \
      --start_index 1 \
      --embedding_layer -1 \
      --embedding_pooling last \
      --no_skip_existing 2>&1 | tee "$eval_log"
  else
    echo "=== SKIP EVAL seed=$seed variant=$variant ==="
  fi
}

for seed in $SEEDS; do
  for variant in $VARIANTS; do
    run_variant "$seed" "$variant"
  done
done
