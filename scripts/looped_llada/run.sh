#!/usr/bin/env bash
# V2.1-a (feedback-bridge): P1 main experiment, frozen R baseline.
#
# RecursiveLink applied ONLY on feedback transitions (r >= 1) -- first
# pass through R is exactly vanilla. T_rec=1 is therefore an architectural
# drop-in to vanilla split LLaDA (no link gradient at T=1, eval-only).
#
# Training T_rec ~ Uniform{2..6}; eval can be any T_rec >= 1.
# Trainable surface: ONLY recursive_link.* (~33M params, entire backbone
# frozen). For the R-unfrozen variant, see run_unfrozen_r.sh (P2a).
#
# Run from the repo root: `bash scripts/looped_llada/run.sh`

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

accelerate launch \
  --config_file scripts/accelerate_configs/zero2.yaml \
  examples/llada_looped/sft.py \
  --output_dir .models/loop_belief/openmath2-v2.1a \
  --dataset_args ".data/sft/llada/openmath2-500k" \
  --load_preprocessed_data True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_length 1024 \
  --learning_rate 5e-4 \
  --report_to tensorboard \
  --logging_dir .models/loop_belief/openmath2-v2.1a/tb \
  --save_steps 0.05 \
  --save_total_limit 3 \
  --save_only_model False \
  --overwrite_output_dir False \
  --eval_strategy no \
  --use_latent_feedback True \
  --t_rec_min 2 --t_rec_max 6 --t_rec_eval 4 \
  --mu_rec_eval 4 \
  --loop_lr_mult 1.0 \
  --diag_log_every 50
