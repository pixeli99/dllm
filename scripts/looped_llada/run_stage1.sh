#!/usr/bin/env bash
# V2 T_rec=1 warmup: trains RecursiveLink + the unfrozen base surface with
# a single recurrent sweep before running the main T_rec-randomized experiment.
# Run from the repo root: `bash scripts/looped_llada/run_stage1.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2-stage1 \
  --dataset_args "nvidia/OpenMathInstruct-2[train:50000,test:5000]" \
  --num_train_epochs 0.5 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 2e-5 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2-stage1/tb \
  --save_steps 0.2 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --t_rec_min 1 --t_rec_max 1 --t_rec_eval 1 \
  --mu_rec_eval 1 \
  --loop_lr_mult 10.0 \
  --diag_log_every 50
