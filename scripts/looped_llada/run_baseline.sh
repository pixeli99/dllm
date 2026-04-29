#!/usr/bin/env bash
# V0 (vanilla LLaDA baseline, FULL-param finetune): same data + training
# config as V1/V2, no loop module. Used as the "data-alone" upper bound.
#
# Note: V1/V2 freeze prelude+coda by default, so this baseline has a
# different trainable surface. For a SURFACE-MATCHED V0 (= V1 with
# T_rec=1, prelude+coda frozen), use:
#   bash scripts/looped_llada/run_v1.sh
# but with `--t_rec_min 1 --t_rec_max 1`.
#
# Run from the repo root: `bash scripts/looped_llada/run_baseline.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada/sft.py \
  --output_dir .models/loop_belief/openmath2-v0-baseline \
  --dataset_args "nvidia/OpenMathInstruct-2[train:500000,test:5000]" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v0-baseline/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no
